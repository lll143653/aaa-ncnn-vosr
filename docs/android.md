# Android 构建与使用

手机端使用官方 VOSR 0.5B 单步模型，与原桌面 1.4B 模型分开。应用为 ARM64、Android 10+；NPU 能力检查限定 SM8550 / HTP v73，已验证设备为小米 13。GPU 可枚举 Vulkan 设备，但目前也只在这台手机完成测试。

## 安装和模型

本地 `deliverables/` 中有三个文件，大文件没有提交到 Git，也还没有发布 GitHub Release：

| 文件 | 用途 |
| --- | --- |
| `vosr-restore-debug.apk` | 含两个推理后端的开发版应用 |
| `vosr-0.5b-htp-v73-models.zip` | NPU 模型包 |
| `vosr-0.5b-ncnn-vulkan-models.zip` | GPU 模型包 |

安装 APK，在“推理后端”选择 NPU 或 GPU，导入对应 ZIP，然后选择照片和放大倍数。两个 ZIP 不能混用。处理结果可对比并保存到相册 `Pictures/VOSR`。小米 13 上建议使用 NPU。

应用会保留原图尺寸，分块处理。输出越大，patch 越多，耗时也越长。生成式复原可能补出原图中无法确认的细节。

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

安装使用 `--no-streaming`。`-SkipModels` 保留手机中已有模型；首次可在应用内导入 ZIP。若本机已有经过验证的 `artifacts/qnn` 和 `artifacts/ncnn-0.5b`，也可去掉 `-SkipModels`，由脚本校验并复制模型。

## 从官方权重转换

只使用现成模型包时可以跳过本节。转换依赖为 `conversion/requirements-mobile.txt`，官方下载位置、文件大小和哈希为 `conversion/mobile-model-downloads.json`。

```powershell
.\scripts\setup-mobile.ps1 -WithConversion
. .\scripts\env.ps1
.\.venv\Scripts\python.exe conversion\export.py --size 512
.\.venv\Scripts\python.exe conversion\export_ncnn.py --reference artifacts\reference
.\.venv\Scripts\python.exe conversion\validate_ncnn.py --reference artifacts\reference --onnx artifacts\onnx
```

第一步下载固定版本的官方源码及 0.5B 权重。ONNX 导出同时生成参考张量并与 PyTorch 对照；ncnn 导出使用 pnnx 20261020，再用 ncnn 20260526 CPU 运行完整链路做数值检查。转换通过不代表手机 Vulkan 已通过，仍需真机回归。

NPU 转换使用 `conversion/build_qnn.py`。先按 QAIRT 2.44 的要求配置其 Python 环境，再加载 `scripts/qnn-env.ps1`，用该环境的 Python 执行转换。它读取 `artifacts/onnx/`，生成 HTP v73 FP16 context；`conversion/validate_device.py` 用 qnn-net-run 对照 ONNX，并在通过后生成 NPU manifest。验证前需要在手机 `/data/local/tmp/vosr/qnn` 准备 SDK 的 qnn-net-run 和 HTP 运行库；多设备连接时用 `ANDROID_SERIAL` 选择设备。不要把普通 PyTorch 环境当作已安装 QAIRT 转换器依赖的环境。

移动端共 12 个子图：VAE encoder、DINO、DiT prepare、7 组四层 DiT、DiT finish、LightDecoder。模型清单保存在 `models/mobile-0.5b/`，大权重与中间文件保存在被 Git 忽略的 `artifacts/`。两个后端的转换和数值边界见 [阶段 8](08-repository-integration.md)。

## 测试和 Profile

```powershell
. .\scripts\env.ps1
Push-Location app
flutter analyze
flutter test
Pop-Location
.\.venv\Scripts\python.exe scripts\test-mobile-backends.py --serial <设备序列号>
```

真机脚本使用固定输入，检查原始尺寸、EXIF、三块拼接、会话复用、后端切换、输出像素和相册副本。它会在手机里执行复原并写入测试图片，不只是读取设备信息。两个后端的原始验收结果见 [阶段 9](09-mobile-backend-acceptance.md)。

Profile 的采集、普通计时与详细事件、前台/休眠分组见 [阶段 10](10-performance-profile.md)。`profile-mobile.py` 需要先用插桩前 APK 采集 `--baseline-only` 基准，再安装当前版本。使用新的 `--output-dir` 保存另一轮采样，避免覆盖已有结果。`analyze-mobile-profile.py` 和 `verify-mobile-profile.py` 默认针对阶段 10 的本地证据目录；后者是这组固定实验的验收脚本。

原始日志和 Perfetto trace 较大，留在本地 `artifacts/`、`evidence/`，不在 Git 中。仓库保留固定回归图片、模型清单、测试报告和采集脚本。`package-mobile.py --apk-only` 会验证已有模型 ZIP 并更新 APK 校验和，不重新打包模型。

## 已知限制

- NPU context 针对 SM8550 / HTP v73，其它高通芯片需要重新转换和验证。
- GPU 为兼容配置：关闭 Winograd、subgroup、cooperative matrix；134 个 MatMul、85 个 Reduction 在 CPU 执行，子图间也存在 CPU Mat 传递。当前速度不适合用来替代小米 13 上的 NPU 路径。
- ncnn 20261020 的 Vulkan 数值测试、旧 pnnx 的 DiT 转换没有通过，因此运行库和转换器采用不同版本，不能只改版本号就视作完成升级。
- 当前使用开发签名。完整发布流程、更多机型和长时间温控测试尚未完成。
