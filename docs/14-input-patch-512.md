# 阶段 14：按输入尺度切 512 / 96

日期：2026-11-15。

这里的 512 和 96 都指缩放后输入图像的像素。先把最长边等比例调整到 832，再按 512×512、重叠 96、步长 416 切块。每块进行真实 2× 推理，输出 1024×1024；在输出坐标中，重叠是 192、步长是 832。拼接后缩回原图尺寸，通过五层高斯金字塔保留输入低频、融合复原细节。

| 工作输入 | 输入块数 | 拼接后的 2× 图像 |
| --- | ---: | --- |
| 832×624 | 2×2 = 4 | 1664×1248 |
| 624×832 | 2×2 = 4 | 1248×1664 |
| 832×416 | 2×1 = 2 | 1664×832 |

末端不足一块时沿边界补齐，输出只保留有效区域。旧实现的模型输入、输出都是 512 RGB；做 2× 时一块只覆盖 256 个 LR 像素，所以会得到 12 块。本次改为 1024 RGB 的完整模型规格，每块确实覆盖 512 LR，不能只修改界面上的数字。

相册选图和拍照都默认使用这套流程，界面仅提供固定 2×。相册不再默认对数千万像素原图做 4× 分块。系统拍照入口仍严格检查原始 `DigitalZoomRatio >= 10`，缺失就拒绝；普通相册选图不增加相机专用门槛。小米相机标准变焦字段缺失的问题见阶段 13。

## 转换

沿用同一套官方 VOSR 0.5B 权重，没有重新训练。VAE RGB 为 1024、latent 为 128，DiT 为 4096 token、64×64 网格；RoPE 使用官方大尺寸位置缩放，DINO 保持 448 输入。

1024 导出遇到两个问题，均在转换时处理：

- ONNX Runtime 对数百万元素一次归约的 GroupNorm 有明显误差。把均值、方差的归约拆成每组最多 256 个元素后，编码器相对 PyTorch 的 NRMSE 从约 1.3e-3 降到约 3.4e-7，没有放宽原来的 1e-3 门槛。
- VAE 和 DiT 的完整注意力矩阵过大。只沿 query 轴分成 256 一组，每组仍访问完整 key/value，保留全局注意力。改写前后对照 PyTorch，再分别导出 QNN 和 ncnn。

QNN 的 VAE 单独保留原生 GroupNorm。分段归约适合稳定的 ORT 参考，却会阻止 HTP 融合，编码器编译曾占用约 20 GB 提交内存。原生版本约 59 秒完成编译；其 ORT CPU 误差只作为诊断记录，不冒充验收结果。真正的验收使用手机 HTP 输出对照稳定的 FP32 ONNX 参考。其余十个子图共用已验证的 ONNX。

导出结果放在独立的 `artifacts/onnx-1024`、`reference-1024`、`qnn-1024` 和 `ncnn-0.5b-1024`。旧 512 规格保留在原目录，避免把不兼容的模型覆盖成相同文件。

```powershell
. .\scripts\env.ps1
.\.venv\Scripts\python.exe conversion\export.py --size 1024 --input-scale 2 --output artifacts\onnx-1024 --reference artifacts\reference-1024
.\.venv\Scripts\python.exe conversion\export_ncnn.py --size 1024 --reference artifacts\reference-1024 --output artifacts\ncnn-0.5b-1024
.\.venv\Scripts\python.exe conversion\validate_ncnn.py --reference artifacts\reference-1024 --onnx artifacts\onnx-1024 --models artifacts\ncnn-0.5b-1024
.\.venv\Scripts\python.exe conversion\export_qnn_large.py --reference artifacts\reference-1024 --output artifacts\onnx-qnn-1024
# 使用已经配置好 QAIRT 依赖的 Python 环境
. .\scripts\qnn-env.ps1
E:\work\ai\vosr-mobile\.venv-qnn\Scripts\python.exe conversion\build_qnn.py --onnx artifacts\onnx-1024 --native-vae-onnx artifacts\onnx-qnn-1024 --output artifacts\qnn-1024 --workers 1
```

QNN 仍为 HTP v73 FP16 计算、FP32 IO；GPU 仍为 FP16 权重、FP32 兼容计算。应用根据 manifest 的 `tileSize` 分配 RGB、noise 和输出缓冲区，导入时拒绝错误后端或旧 512 输出模型。

## 手机内存限制

小米 13 的 Vulkan `maxStorageBufferRange` 实测为 134217728 字节，即 128 MiB。1024 VAE 的部分特征图为 `128×1024×1024×4` 字节，即 512 MiB。提交这类缓冲区后，输出严重偏离参考；关闭 Winograd、SGEMM 或把 GroupNorm、Softmax 回退 CPU 都不能修复。因此新规格的 GPU 兼容配置将整个 VAE 编码器、解码器放在 CPU，DINO、DiT 继续走 ncnn Vulkan 及原有 CPU 算子回退。

NPU 单子图执行通过，但初版一次加载 12 个大尺寸 context 时触发了 DSP 重置。新加载流程先读取所有子图的 spill/fill 工作区需求，再通过 HTP 多 context 分组共享最大工作区，实测最大需求为 585170944 字节。仅共享工作区仍不够，因此常驻 encoder、DINO、prepare、finish、decoder 五个 context，七组 DiT 按顺序加载、执行后释放，最多同时保留六个。权重和 IO 不共用，DiT 双缓冲顺序不变。每块的 `contextLoadMs`、`contextFreeMs` 单独记录，`nativePatchMs` 包含这部分开销。

GPU 的 CPU VAE 保留关闭 SGEMM。公开参考输入上，普通 CPU 卷积编码器约 114 秒，SGEMM 对照约 171 秒且有大量换页；其并行测试曾使应用在融合阶段退出。最终验收不再并行运行手机上的内存密集测试。

GPU 的 12 个网络同时保留权重时，CPU 解码阶段也曾退出。因此 1024 规格按子图加载、执行后释放，最多保留一个网络。阶段间传递 CPU Mat，另外记录 `streamGraphs`、`maxResidentNetCount`、`netLoadMs`，加载开销计入原生总耗时。旧 512 规格继续使用原来的常驻模式。

## 验证状态

12 个 ONNX 子图已通过 PyTorch 数值检查，拆分 DiT 对官方完整调用的最大绝对差为 5.72e-6。Flutter 的 4 项回归通过，Android APK 已构建并安装。真机 `CameraChecks` 已通过输入块数、八种 EXIF 方向、严格变焦门槛和金字塔融合检查，分块与整幅融合的差异为 0 LSB。

桌面 ncnn 完整链路通过，输出相对 FP32 参考为 79.68 dB；手机 QNN 单图顺序执行及实际 JNI 输出均为 60.16 dB。应用内四块流程已通过：4000×3000 输入，832×624 工作图，4 次完整模型调用，结果融合回 4000×3000 并保存到相册。记录中 `nativePatchMs` 长度为 4、`patchRunCount=4`，共执行 48 次子图；context 创建次数 33 是模型加载次数，与图片块数不同。

相机分支的三个合成 EXIF 用例也通过：缺失变焦、9× 被拒绝，12× 且方向为 6 的 1001×751 原片得到 751×1001 输出，工作图 624×832，同样为 4 块。小米系统相机真实原片仍缺少标准变焦字段，会按严格规则拒绝。

| NPU 12MP 样例 | 测量值 |
| --- | ---: |
| 四块原生调用 | 10.826 / 11.176 / 11.150 / 11.189 秒 |
| 四块原生调用合计 | 44.34 秒 |
| 区域读取和双三次采样合计 | 12.37 秒 |
| DINO 预处理合计 | 4.54 秒 |
| 拼接及 PNG 写出 | 6.03 秒 |
| 原尺寸金字塔融合 | 16.83 秒 |
| 从准备到得到结果 | 86.96 秒 |

原生调用包含 DiT context 的加载和释放。末块的 12 次 `graphExecute` 合计约 7.03 秒，加载约 4.04 秒、释放约 0.09 秒。1024 输出规格的计算量与内存需求都增加了，4 块不代表会比旧模型的 12 小块更快。

GPU 兼容路径的完整 JNI 调用也已通过，输出相对 FP32 参考为 **68.96 dB**，单块原生调用约 **343.25 秒**，包括每个子图的加载；最多同时保留一个网络。四块拼接、原尺寸融合及保存的公共流程使用 NPU 验证，没有另跑 GPU 四块整图。

这些是单次功能验收的计时，未控制温度、频率和休眠状态，不作为前台稳定性能基准。512 输出规格的 1.588 秒/块属于历史测量。新规格没有降低模型计算量，当前小米 13 推荐使用 NPU。数值和计时摘要保存在 [验收数据](benchmarks/input-patch-512-2026-11-15.json)。

## 交付

修正版 APK 已安装，新规格模型已放在手机的 `models-1024`、`models-gpu-1024`。本地 `deliverables` 提供独立的 `vosr-0.5b-htp-v73-1024-models.zip`、`vosr-0.5b-ncnn-vulkan-1024-models.zip`，文件大小和 SHA-256 由打包脚本检查并写入 `SHA256.json`。旧模型目录和包继续保留，不能导入新流程。

本轮只读取公开参考张量、合成 EXIF 测试图及必要日志，没有截屏或读取私人照片像素。所有源码、工具、模型及转换产物都在 `E:\work\ai`。新 1024 规格沿用 512 训练权重，只完成数值与功能检查，尚未做更多实拍画质评测。
