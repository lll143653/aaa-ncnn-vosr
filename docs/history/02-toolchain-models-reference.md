# 阶段 2：工具链、官方权重与参考推理

完成日期：2026-11-14（Asia/Shanghai）。

## 已完成的工作

- 在 `E:\work\ai\tools` 安装便携式 uv、独立 CPython 3.11 / 3.10、Flutter stable 3.50.0 / Dart 3.14.0、Temurin JDK 17.0.21、Android 命令行工具、platform-tools 37.0.1、Android 36 平台与 build-tools 36.0.0。
- 从 Qualcomm 官方 SDK 按 HTTP 字节范围仅提取必要工具和运行库。版本为 QAIRT 2.44.0.260225，包含 Windows 转换工具、QNN 头文件、HTP v73 Android 运行库。ZIP 文件解压校验 CRC。
- Python 推理环境为 `.venv`（PyTorch 2.5.1 CPU / torchvision 0.20.1、diffusers 0.35.0）；QNN 转换使用单独的 `.venv-qnn` / Python 3.10，避免原生扩展 ABI 冲突。
- 官方源码提交：`516f292b99cf23c76fdc33351e86dc4f97711fe8`。
- 官方模型提交：`f24b3061b7f350b81e1907bdcfc27f71fb4ff3f3`。
- VOSR 0.5B 单步主权重已完整下载（1,948,337,520 字节），SHA-256：`45b6687b4ce59fa3451300f9e0fdcaf2e729c15e65f444af210a7851dfa13203`。
- 配套 VAE、DINOv2、轻量解码器已下载并验证官方 SHA-256；下载清单见 `evidence/model-downloads.json`。

## 参考推理

使用官方示例 `Nikon_024_LR4.png` 中间的 128×128 区域，以 4× 复原为 512×512。使用固定噪声、posterior mode、单步 t=1→0，FP32 CPU 推理，不启用颜色校正。

- 原始输入：`artifacts/reference/input.png`。
- 双三次参考：`artifacts/reference/bicubic.png`。
- VOSR 结果：`artifacts/reference/restored.png`。
- 参数及完整分图与原始未拆分 DiT 的差异：`artifacts/reference/pipeline.json`。
- 完整拆分链路对原始未拆分 DiT 的最大绝对误差：`1.1444091796875e-05`（latent）。

所有 12 个 ONNX 子图均通过 ONNX checker 和 ONNX Runtime 与 PyTorch 的数值对比。每个子图的形状、NRMSE、最大绝对误差、文件 SHA-256 保存在 `artifacts/onnx/*.validation.json`。

## 发现并修正的问题

1. 256×256 输出虽可数值复现，但在示例上出现明显伪影。正式图改用训练分辨率 512×512；早期实验保存在 `artifacts/baseline256`。不能将 256 的数值正确当成图像质量合格。
2. GitHub 直连和 Hugging Face 大文件连接多次中断。修复断点保留逻辑，并采用经过 Content-Range 校验的并发下载和本机已有代理。最终以完整 SHA-256 为准。
3. 旧 ADB 推送时会截断文件名末尾字符。已改用工作目录中下载的官方 platform-tools ADB。
4. 最新 Android CLI 使用 `platforms/android-36` 等包名；兼容 sdkmanager 脚本会错误拆分分号包名。已改用 `android.exe sdk install`。
5. QNN Python 原生扩展需要 Python 3.10；不能使用推理环境中的 Python 3.11。
6. QNN 转换器同时要求源布局与目标布局，固定 NCHW / NTF 及 FP32 输入输出，内部计算使用 FP16。

## 复现

```powershell
. E:\work\ai\vosr-mobile\scripts\env.ps1
E:\work\ai\vosr-mobile\.venv\Scripts\python.exe E:\work\ai\vosr-mobile\conversion\export.py --size 512
```

该命令会重新导出 12 个子图并做数值校验，需要数 GB 空闲内存与磁盘空间。缓存、临时文件和依赖安装路径由 `scripts/env.ps1` 设置在工作目录内。

## 本阶段边界

本阶段已完成桌面参考推理与 ONNX 数值验证。完整模型 NPU 执行、Flutter APK 编译和应用内完整处理尚在后续阶段；解码器的单独 NPU 成功不能替代完整链路验收。

## 1.4B 版本的初步评估

官方清单中的 VOSR2 主权重为 5,576,385,424 字节，比首版 0.5B 权重更大；使用 16 通道 Qwen-Image VAE（配套权重 178,418,892 字节），而首版使用 4 通道 SD2.1 VAE。不能把 1.4B 权重直接放入当前 0.5B 应用模型包替换使用。

适配需要重新导出 Qwen VAE、16 通道 latent 的 DiT 子图、重新编译 HTP context，并单独测量峰值内存、精度和耗时。本次将 0.5B 作为手机首版；没有下载和实测 1.4B，因此不声称其在小米 13 上已兼容或达到某种性能。
