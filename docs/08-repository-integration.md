# 阶段 8：融合 aaa-ncnn-vosr 与双后端模型转换

日期：2026-11-14。该阶段已完成仓库内源码融合、双后端接口、APK 构建及完整 ncnn 数值校验；手机 GPU 验收单列阶段 9。

后续更新：本阶段初始采用 20261020；真机验证发现该版本 Vulkan 数值异常。阶段 9 将运行库固定为原项目的 ncnn 20260526 / e54f7b1，转换器保留 pnnx 20261020，并禁用未通过数值对照的 Winograd 卷积路径。最终精度以阶段 9 为准。

## 基础与范围

克隆用户指定的 `https://github.com/lll143653/aaa-ncnn-vosr`，基线提交 `26e8a46`，本地分支 `feat/mobile-npu-gpu`。工作目录 `E:\work\ai\aaa-ncnn-vosr`。原项目的 `cpp/`、桌面脚本与 1.4B 参数文件保留；新增手机端路径，没有把 1.4B 的模型包标成 0.5B。

原项目使用 Qwen VAE、DINOv2-L、36 层 1.4B DiT；当前 NPU 工作使用 SD2.1 VAE、DINOv2-B、28 层 0.5B DiT 和 LightDecoder。为使手机 NPU / GPU 可比，移动端两个后端统一使用官方 `VOSR_0.5B_os` 权重，原桌面 1.4B 路径本次未重新实测。

## 应用融合

- `app/`：迁入已验证的 Flutter 照片选择、原图对比、保存、原始分辨率输入、EXIF、重叠 patch 和流式 PNG；新增后端切换与分别导入模型包。
- `NativeQnn.kt` / `vosr_jni.cpp`：保留常驻 12 个 QNN HTP context、内存张量传递、取消与逐图计时。
- `NativeGpu.kt` / `ncnn_jni.cpp`：ncnn Vulkan 常驻 12 个网络，使用 ncnn 对齐内存，原生数组输入输出，不生成子进程或子图间文件。网络间目前使用 CPU Mat，经 ncnn 上传/下载，尚未跨网络全程保留 VkMat。
- GPU 强制 FP32 计算和存储缓冲区，模型文件保留 FP16 权重。ncnn 可对少数不支持 Vulkan 的算子在 CPU 执行；没有无 GPU 时整条改为 CPU 的回退路径。
- 一个常驻会话管理器按后端、模型目录、manifest 时间戳复用；切换时释放之前的引擎。NPU 模型目录沿用 `models`，GPU 使用 `models-gpu`，保留当前手机上的模型安装。
- NPU 界面能力检查限定 SM8550；GPU 使用 Vulkan 枚举。模型包分别检查后端、0.5B 类型、块尺寸、子图数量和文件 SHA-256；ZIP 禁止路径穿越和重复文件，限制解压总量，替换前完整校验。

输出 patch 仍为 512×512，步长 448、重叠 64，高斯融合。2× / 4× 输入都读取原始区域，保持整图原尺寸，预览缩略图不用于模型输入。只保留当前和上一行最终 patch 缓存，逐行写 PNG。

## 同权重 GPU 转换

新增 `conversion/export_ncnn.py`，复用官方模型加载与相同 12 个边界：encoder、DINO、prepare、7 组各 4 层 DiT、finish、decoder。固定 512 输出块、4×64×64 latent、1024×1024 token。

转换改写包括：QKV 分开后再 reshape，避免 5D；RoPE 的偶奇旋转改为常量置换矩阵；finish 采用通道置换和 PixelShuffle，避免 6D；去除 FP32 路径中冗余 dtype cast；VAE 的 baddbmm 分解为矩阵乘、缩放和 softmax。每次先与原 PyTorch 模块比较，再 trace 和 pnnx，并拒绝残留自定义算子。DiT 及 finish 的 PyTorch 改写最大差为 0，encoder 约 1.43e-6，DINO 融合 SDPA 与展开版本最大差约 1.53e-4。

`conversion/validate_ncnn.py` 在桌面运行实际 ncnn CPU 全链路，并对每个边界与 FP32 ONNX 比较。12 个子图通过，逐图 NRMSE 小于 0.00118；完整输出浮点像素相对独立 FP32 ONNX 链路 PSNR **77.6168 dB**，使用 [-1,1] 的范围 2 计算。该结果验证转换，不是复原质量评分，也不是 GPU 真机通过的替代。

证据：`artifacts/ncnn-0.5b/validation.json`、各图 `.validation.json`、`evidence/ncnn-numerical-validation.log`、各图 pnnx 日志。前期失败探测日志保留用于诊断；最终模型以 `manifest.json` 和阶段 9 的发布校验清单为准。

当前 `artifacts/ncnn-0.5b/validation.json` 已更新为阶段 9 的最终运行库检查（76.6105 dB）；本阶段初始 77.6168 dB 结果保留在 `evidence/ncnn-numerical-validation.log`。

## 构建与复现

`scripts/build-native.ps1` 构建独立 HTP / Vulkan JNI 库，`build-android.ps1` 随后构建 APK。ncnn 固定 20261020 / `7e36d25`，官方 Android Vulkan SDK 下载包 SHA-256：`6e0c5f9b397faf32b923de2ef6210af9bf375851869dceff06a8005331981086`。ncnn 官方该包使用 NDK r30；与 r28 的 C++ 运行库混用会缺少 `std::__ndk1::__hash_memory`，已定位并在阶段 9 处理。

Flutter 静态检查无问题，APK 构建记录为 `evidence/dual-package-retry.log`。迁入后 NPU 首次样例 3 patch 总耗时 10.218 秒、流水线 7.182 秒，输出与此前原始分辨率版本逐像素一致。后续重复样例的耗时受运行状态影响，阶段 9 记录最终切换测试。

官方源码固定 `cswry/VOSR@516f292b99cf23c76fdc33351e86dc4f97711fe8`；官方 0.5B 权重清单为 `conversion/mobile-model-downloads.json`；精确 Python 依赖为 `conversion/requirements-mobile.txt`。

```powershell
.\scripts\setup-mobile.ps1 -WithConversion
.\.venv\Scripts\python.exe conversion\export.py --size 512
.\.venv\Scripts\python.exe conversion\export_ncnn.py --reference artifacts\reference
.\.venv\Scripts\python.exe conversion\validate_ncnn.py --reference artifacts\reference --onnx artifacts\onnx
.\scripts\build-android.ps1
```

NPU 转换继续使用 `conversion/build_qnn.py`，需要 QAIRT 专用 Python 环境与 `scripts/qnn-env.ps1`。已验证的 HTP 模型包可直接复用，无须重复转换。所有工具、缓存和模型继续放在 `E:\work\ai`。按用户要求，本阶段未截屏或读取手机屏幕。
