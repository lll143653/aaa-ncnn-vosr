# 阶段 9：双后端真机验收与交付

完成日期：2026-11-14（Asia/Shanghai）。融合版已安装到小米 13，NPU → GPU → GPU 复用 → NPU 的连续验收全部通过。验收只使用应用日志、模型输入输出文件、哈希和 MediaStore 元数据，未截屏或读取手机屏幕。

## 最终实现

手机端 NPU / GPU 均使用官方 VOSR 0.5B 单步权重，共享同一份原始输入、EXIF 区域解码、2× / 4×、重叠 patch、高斯融合和流式 PNG。原桌面 VOSR2 1.4B 路径保留，本次未重新实测。

- NPU：Qualcomm QAIRT 2.44 / HTP v73，12 个常驻 context，内部 FP16、IO FP32。新增执行期间的独立 power client 投票，结束或取消时恢复空闲 DCVS 配置，释放引擎时销毁该 client。SDK 实际返回支持且接受投票。
- GPU：ncnn **20260526 / e54f7b1** 运行库，pnnx **20261020** 转换器，FP16 权重文件、FP32 计算。实际设备为 Adreno 740；12 个网络常驻并跨图片复用。
- GPU 兼容配置：关闭 Winograd 和 subgroup 优化，将 **85 个 Reduction 层**显式交由 CPU FP32 执行，其余可支持的算子走 Vulkan。子图间通过内存传递，无子进程和子图张量文件；最终 patch 行缓存仍用于有界内存拼接。
- 后端切换释放旧引擎的模型和 context；未调用原桌面 demo 的 `_exit`。模型目录、校验和及 UI 就绪状态分别管理。

这些配置来自真机对照：20261020 Android Vulkan 运行库未通过首层卷积数值检查；20260526 转换器未通过 DiT 检查；旧运行库的 Winograd 有明显偏差，4D Reduction Vulkan pipeline 会崩溃。最终组合通过完整 ncnn CPU / ONNX 校验，以及下面的应用内 GPU 输出校验。相关探针源码为 `cpp/tools/probe_gpu.cpp`，历史失败日志保留在本地 `evidence/`，不作为成功证据。

## 数值与图像检查

最终组合的 12 个 ncnn 子图，在桌面实际执行并对照独立 FP32 ONNX 链路：输出浮点像素 PSNR **76.6105 dB**，RMSE **0.000295464**，范围 [-1,1]。证据：`artifacts/ncnn-0.5b/validation.json`、`evidence/ncnn-final-runtime-validation.log`。这是转换检查，不代表复原质量评分。

手机应用运行于普通应用 UID。两个固定样例均为 2×、3 个 patch：PNG 原始逻辑尺寸 514×129，输出 1028×258；JPEG 编码尺寸 514×129、EXIF=6，逻辑尺寸 129×514，输出 258×1028。`inputDownscaled=false`、`streamingOutput=true`，无全图预缩小。

| 连续运行顺序 | 整次耗时 | 处理流水线 | 会话加载 | 输出对比 |
| --- | --- | --- | --- | --- |
| NPU，PNG，首次加载 | 12.042 秒 | 7.395 秒 | 2.837 秒 | 与此前 NPU 基准逐像素一致 |
| GPU，PNG，切换后首次加载 | 159.059 秒 | 133.910 秒 | 23.426 秒 | 对 NPU 基准 PSNR 53.7597 dB |
| GPU，EXIF JPEG，复用会话 | 133.185 秒 | 133.181 秒 | 0 秒 | 对 NPU 基准 PSNR 51.0567 dB |
| NPU，EXIF JPEG，切回后加载 | 11.735 秒 | 6.744 秒 | 2.145 秒 | 与此前 NPU 基准逐像素一致 |

“整次耗时”包含本次模型校验、会话准备及处理，不包含之后的相册保存。“处理流水线”包含预处理、逐块推理和拼接。“单块推理”从原生调用计时，包含完整 VAE 编码、DINO、DiT、解码及输入输出拷贝，不含模型首次加载和图像预处理。

本轮 6 个 NPU patch 的单块推理为 **1.593–1.613 秒**；6 个 GPU patch 为 **43.449–44.236 秒**。GPU 第二张的网络创建数仍为 12、累计 patch 数为 6，证实跨图片复用。切回 NPU 后重新创建 12 个 context，`boostVoteAccepted=true`，输出没有变化。

四组均取得真实 MediaStore URI，`is_pending=0`，相册副本与输出 PNG 的 SHA-256 完全一致。原始尺寸、EXIF、块数、输出有限数值和 PNG 内容检查均通过。GPU 与 NPU 并非逐像素相同，上表 PSNR 是同输入输出的一致性检查。

本轮退出 GPU 并切回 NPU 后，应用内存快照 PSS **1,759,275 KB（约 1.68 GiB）**；该值包含已初始化的 GPU 设备缓存，不是 GPU 推理峰值。当前兼容 GPU 路径明显慢于 NPU；在小米 13 上建议使用 NPU。结果为单次连续功能验收，没有进行稳定温度、多轮频率控制的性能统计；其它 Vulkan 手机和低内存设备尚未验证。

## 构建、证据与交付

- Flutter 静态检查无问题；后端选择测试通过，覆盖模型就绪状态隔离和导入参数，见 `evidence/dual-analyze.log`、`evidence/backend-selector-test.log`。
- 原生代码和 Android APK 构建成功，见 `evidence/gpu-reduction-compat-build.log`。QNN JNI 使用 NDK r28，GPU JNI 使用 r30，并在链接时拒绝未解析符号。
- 最终交付另在界面标明 GPU 兼容模式较慢，构建见 `evidence/final-delivery-build.log`；最终 APK 的 6 个原生库与完成上述真机验收的版本逐字节一致。
- 结构化真机结果、每例完整结果、输出及相册副本：`artifacts/mobile-tests/acceptance.json` 及其同级样例目录；日志：`evidence/dual-backend-final-acceptance.log`。
- APK v2 签名验证通过；APK 中 6 个原生库与构建输入一致，见 `evidence/final-apk-signature.txt`、`evidence/apk-native-verification.json`。
- APK：`deliverables/vosr-restore-debug.apk`。离线模型包：`vosr-0.5b-htp-v73-models.zip`（1,257,347,674 字节）、`vosr-0.5b-ncnn-vulkan-models.zip`（1,182,903,112 字节）。完整 SHA-256 见 `deliverables/SHA256.json`。
- NPU ZIP 与此前交付 SHA-256 完全相同，可继续使用。GPU ZIP 经完整文件哈希检查后打包。模型 manifest 保存在 `models/mobile-0.5b/{npu,gpu}/`，大模型、APK、SDK 二进制不纳入 Git。

```powershell
.\scripts\setup-mobile.ps1
.\scripts\build-android.ps1
.\scripts\install-android.ps1 -Backend both
.\.venv\Scripts\python.exe scripts\test-mobile-backends.py
.\.venv\Scripts\python.exe scripts\package-mobile.py
```

本地开发分支为 `feat/mobile-npu-gpu`。代码尚未推送 GitHub。此轮没有进行系统文件选择器和滑动对比控件的真人点击回归，已有后端选择组件测试和应用内文件级验收；未通过屏幕访问进行检查。
