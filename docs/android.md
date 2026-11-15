# Android 构建与使用

手机端使用官方 VOSR 0.5B 单步模型，与原桌面 1.4B 模型分开。应用为 ARM64、Android 10+。已验证小米 13（SM8550 / HTP v73 / Adreno 740）和 OPPO PMA110（SM8850 / HTP v81 / Adreno 840）；NPU 模型按芯片区分，GPU 按设备缓冲区上限选择 VAE 路径。

## 安装和模型

本地 `deliverables/` 包含下列文件，大文件没有提交到 Git，也还没有发布 GitHub Release：

| 文件 | 用途 |
| --- | --- |
| `vosr-restore-debug.apk` | 含两个推理后端的开发版应用 |
| `vosr-0.5b-htp-v73-1024-models.zip` | NPU 模型包，512 LR 输入块 / 1024 输出块 |
| `vosr-0.5b-ncnn-vulkan-1024-models.zip` | GPU 模型包，同一输入、输出规格 |
| `vosr-0.5b-sm8850-htp-v81-1024-models.zip` | PMA110 的 NPU 模型包，同一规格 |
| `vosr-0.5b-sm8850-ncnn-vulkan-1024-models.zip` | PMA110 已验证的 GPU 模型包，权重相同 |

安装 APK，在“推理后端”选择 NPU 或 GPU，导入对应设备的 1024 ZIP，然后选择照片。不同芯片的 NPU 包、两个后端的 ZIP 不能混用；旧 512 输出模型也不适用于新流程。处理结果可对比并保存到相册 `Pictures/VOSR`。两台已验证手机均建议使用 NPU。

模型就绪后也可点击“拍照并复原（≥10×）”。相机照片按 EXIF 严格校验，使用固定 832 / 96 重叠 / 2× 处理和原尺寸金字塔融合，具体规则见 [拍照复原](12-camera-restoration.md)。

相册和拍照均先等比例调整到最长边 832，以 512 输入块、96 输入重叠、416 输入步长处理；固定 2×，每块输出 1024。832×624 实际运行 4 次模型推理。拼接后缩回原图尺寸与 LR 做金字塔融合。分块说明和新规格验证见[阶段 14](14-input-patch-512.md)。生成式复原可能补出原图中无法确认的细节。

## 构建环境

当前脚本针对 Windows PowerShell，工作区为 `E:\work\ai`，仓库为 `E:\work\ai\aaa-ncnn-vosr`。以下基础工具需要事先准备；`setup-mobile.ps1` 不是从空白系统安装全部工具链的脚本。

| 工具 | 已验证版本 / 目录 |
| --- | --- |
| Flutter / Dart | 3.50 / 3.14，`tools/flutter` |
| JDK | 17.0.21，`tools/jdk-17.0.21+9` |
| Android SDK | API 36、build-tools 36.0.0，`tools/android-sdk` |
| NDK（QNN JNI） | 28.2.13676358 |
| NDK（GPU JNI） | 30.0.16248370，由准备脚本检查和安装 |
| Qualcomm QAIRT | 2.44.0.260225，`tools/qairt/2.44.0.260225` |
| ncnn Android Vulkan SDK | 20260526，由准备脚本下载并校验 SHA-256 |
| Gradle | Wrapper 固定 9.5.0，并校验发行包 SHA-256 |
| Python / uv | 转换与测试使用 Python 3.11；uv 位于 `tools/uv` |

表中 `tools/` 相对 `E:\work\ai`。QAIRT 需要按 Qualcomm 官方安装和许可要求准备，仓库不附带厂商 SDK。Gradle 首次运行会从官方地址下载；需要代理时，在本机的 `GRADLE_USER_HOME/gradle.properties` 配置代理，不要把个人代理写入仓库。

在仓库根目录执行：

```powershell
.\scripts\setup-mobile.ps1
.\scripts\build-android.ps1
```

准备脚本会安装缺少的 NDK r30、下载 ncnn SDK、从已安装的 QAIRT 复制运行库，生成 `local.properties` 并获取 Flutter 依赖。构建脚本先编译两个 JNI 库，再输出 `deliverables/vosr-restore-debug.apk`。应用依赖固定在 `app/pubspec.lock`。

接入手机后查看 adb 序列号，并显式传给安装脚本。小米手机还需要允许 USB 调试和 USB 安装：

```powershell
E:\work\ai\tools\android-sdk\platform-tools\adb.exe devices
.\scripts\install-android.ps1 -Serial <设备序列号> -SkipModels
```

安装使用 `--no-streaming`。`-SkipModels` 保留手机中已有模型；首次可在应用内导入 ZIP。也可去掉 `-SkipModels`，由脚本按设备 SoC 校验并复制模型：SM8550 使用 `artifacts/qnn-1024` 和 `artifacts/ncnn-0.5b-1024`，SM8850 使用 `artifacts/qnn-sm8850-1024` 和 `artifacts/ncnn-sm8850-1024`。`-TileSize 512` 仅供小米 13 的历史调试流程回归。

## 从官方权重转换

只使用现成模型包时可以跳过本节。转换依赖为 `conversion/requirements-mobile.txt`，官方下载位置、文件大小和哈希为 `conversion/mobile-model-downloads.json`。

```powershell
.\scripts\setup-mobile.ps1 -WithConversion
. .\scripts\env.ps1
.\.venv\Scripts\python.exe conversion\export.py --size 1024 --input-scale 2 --output artifacts\onnx-1024 --reference artifacts\reference-1024
.\.venv\Scripts\python.exe conversion\export_ncnn.py --size 1024 --reference artifacts\reference-1024 --output artifacts\ncnn-0.5b-1024
.\.venv\Scripts\python.exe conversion\validate_ncnn.py --reference artifacts\reference-1024 --onnx artifacts\onnx-1024 --models artifacts\ncnn-0.5b-1024
.\.venv\Scripts\python.exe conversion\export_qnn_large.py --reference artifacts\reference-1024 --output artifacts\onnx-qnn-1024
```

第一步下载固定版本的官方源码及 0.5B 权重。ONNX 导出同时生成参考张量并与 PyTorch 对照；ncnn 导出使用 pnnx 20261020，再用 ncnn 20260526 CPU 运行完整链路做数值检查。转换通过不代表手机 Vulkan 已通过，仍需真机回归。

NPU 转换使用 `conversion/build_qnn.py --onnx artifacts/onnx-1024 --native-vae-onnx artifacts/onnx-qnn-1024 --output artifacts/qnn-1024 --workers 1`。先按 QAIRT 2.44 的要求配置其 Python 环境，再加载 `scripts/qnn-env.ps1`，用该环境的 Python 执行转换。`conversion/validate_device.py --reference artifacts/reference-1024 --onnx artifacts/onnx-1024 --models artifacts/qnn-1024 --output artifacts/device1024` 用 qnn-net-run 对照 ONNX，通过后生成 NPU manifest。验证前需要在手机 `/data/local/tmp/vosr/qnn` 准备 SDK 的 qnn-net-run 和 HTP 运行库；多设备连接时用 `ANDROID_SERIAL` 选择设备。不要把普通 PyTorch 环境当作已安装 QAIRT 转换器依赖的环境。

移动端共 12 个子图：VAE encoder、DINO、DiT prepare、7 组四层 DiT、DiT finish、LightDecoder。模型清单保存在 `models/mobile-0.5b/`，大权重与中间文件保存在被 Git 忽略的 `artifacts/`。两个后端的转换和数值边界见 [阶段 8](08-repository-integration.md)。

SM8850 的编译目标改为 `--soc-model 87 --dsp-arch v81 --vtcm-mb 0 --extended-udma`，需补齐同版本 v81 运行库，完整命令及验证见 [阶段 15](15-sm8850-adaptation.md)。旧 v73 context 不能直接使用。

## 测试和 Profile

```powershell
. .\scripts\env.ps1
Push-Location app
flutter analyze
flutter test
Pop-Location
.\.venv\Scripts\python.exe scripts\test-photo-backends.py --serial <设备序列号> --deploy
.\.venv\Scripts\python.exe scripts\test-camera.py --serial <设备序列号>
```

新规格脚本先通过 JNI 对照公开参考张量，再用固定样图检查两个后端的 4 次真实推理、原尺寸融合和相册保存。相机脚本另外检查严格 EXIF、横竖方向及小图放大。它们会执行复原并写入测试图片，不读取屏幕。`test-mobile-backends.py` 保留阶段 9 的旧 512 输出、原分辨率路径回归，需旧模型。

Profile 的采集、普通计时与详细事件、前台/休眠分组见 [阶段 10](10-performance-profile.md)。`profile-mobile.py` 需要先用插桩前 APK 采集 `--baseline-only` 基准，再安装当前版本。使用新的 `--output-dir` 保存另一轮采样，避免覆盖已有结果。`analyze-mobile-profile.py` 和 `verify-mobile-profile.py` 默认针对阶段 10 的本地证据目录；后者是这组固定实验的验收脚本。

原始日志和 Perfetto trace 较大，留在本地 `artifacts/`、`evidence/`，不在 Git 中。仓库保留固定回归图片、模型清单、测试报告和采集脚本。`package-mobile.py --apk-only` 会验证已有模型 ZIP 并更新 APK 校验和，不重新打包模型。

SM8850 的测试脚本自动使用独立证据目录，打包时添加 `--soc SM8850`。本轮验收运行 NPU 四块整图与 GPU 单次完整 JNI；GPU 测试使用 `--backends gpu --raw-only`。PMA110 的 NPU 12MP 样例为 43.8 秒、GPU 单块约 169 秒，均为未控制温度和频率的功能验收记录，加载波动及范围见阶段 15。

## 已知限制

- NPU context 分别针对 SM8550 / HTP v73 和 SM8850 / HTP v81，其它高通芯片需要重新转换和验证。
- GPU 为兼容配置：关闭 Winograd、subgroup、cooperative matrix；部分 MatMul、Reduction 在 CPU 执行，子图间也存在 CPU Mat 传递。当前速度不适合用来替代小米 13 上的 NPU 路径。
- ncnn 20261020 的 Vulkan 数值测试、旧 pnnx 的 DiT 转换没有通过，因此运行库和转换器采用不同版本，不能只改版本号就视作完成升级。
- 当前使用开发签名。完整发布流程、更多机型和长时间温控测试尚未完成。
