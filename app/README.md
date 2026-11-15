# VOSR Android

Flutter 界面、Kotlin 图像流水线和两个 C++ 推理后端。安装与构建步骤见 [Android 文档](../docs/android.md)。

| 文件 | 职责 |
| --- | --- |
| `lib/main.dart` | 后端选择、模型导入、图片对比、进度与保存 |
| `MainActivity.kt` | Flutter 通道、文件选择和 MediaStore 保存 |
| `NativeImage.kt` | EXIF 方向及原始图像区域读取 |
| `PatchPipeline.kt` / `StreamingPng.kt` | 重叠 patch、行缓存、拼接和 PNG 写入 |
| `PhotoPlan.kt` / `CameraPipeline.kt` | 输入块尺度、832 工作图与原尺寸金字塔融合 |
| `QnnRunner.kt` | 模型包校验与两种后端的调用入口 |
| `NativeQnn.kt` / `NativeGpu.kt` | JNI 接口和会话复用 |
| `vosr_jni.cpp` | QNN HTP context、性能投票及事件采集 |
| `ncnn_jni.cpp` | ncnn Vulkan 网络及 CPU 回退算子计时 |

Kotlin 文件位于 `android/app/src/main/kotlin/ai/vosr/vosr_restore/`，C++ 文件位于 `android/app/src/main/cpp/`。

界面默认将输入等比例调整到最长边 832，按 512 输入块、96 输入重叠、416 输入步长处理；固定 2×，输出块 1024、输出重叠 192。832×624 实际推理 4 次。高斯权重拼接后缩回原尺寸，再与原始 LR 做五层高斯金字塔融合。模型子图间通过内存传递张量，磁盘行缓存用于输出拼接。

拍照入口通过 `CameraPhoto.kt` 严格校验 EXIF 数字变焦 ≥10×，普通相册选图不受此相机门槛限制。之后两者共用处理流程。`photo_jni.cpp` 负责区域重采样和融合，`PhotoNative.kt` 提供 JNI 接口。见[阶段 14](../docs/14-input-patch-512.md)。

模型含 12 个子图。1024 输出规格的 NPU 共享 spill/fill 工作区，常驻 5 个 context，7 组 DiT 顺序加载并释放，最多同时保留 6 个；SM8850 另启用 v81 far mapping。GPU 一次只保留一个 Net，根据实际存储缓冲上限选择 GPU 或 CPU VAE。旧 512 输出模型仍保留全部 12 个常驻会话。切换后端会释放旧会话。阶段 10 的算子数量及计时属于旧 512 输出规格。

模型导入检查芯片、后端、输出规格、文件名、大小限制和 SHA-256。`NpuTarget.kt` 区分 SM8550/v73 和 SM8850/v81，NPU 分别使用 `models-1024`、`models-sm8850-1024`；GPU 使用 `models-gpu-1024`。目录由应用创建，不要用 adb shell 提前创建。旧 `models`、`models-gpu` 保留给历史调试入口。

从仓库根目录执行本地检查：

```powershell
. .\scripts\env.ps1
Push-Location app
flutter analyze
flutter test
Pop-Location
```

`MainActivity` 提供仅在 debug 构建启用的 intent 测试入口。`vosr_test_photo=true` 使用当前流程，`vosr_test_camera=true` 再增加严格变焦校验；不带这两个标志的旧入口保留原始尺寸 2× / 4× 回归。`ModelChecks` 用公开参考张量检查真实 JNI 输出，`CameraChecks` 检查 EXIF、几何和融合。正常界面默认关闭详细 profile。测试通过文件与日志验收，不读取屏幕。

当前交付使用开发签名。发布版需要单独配置签名；不要提交签名文件、`key.properties` 或 `local.properties`。
