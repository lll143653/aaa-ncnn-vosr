# VOSR Android

Flutter 界面、Kotlin 图像流水线和两个 C++ 推理后端。安装与构建步骤见 [Android 文档](../docs/android.md)。

| 文件 | 职责 |
| --- | --- |
| `lib/main.dart` | 后端选择、模型导入、图片对比、进度与保存 |
| `MainActivity.kt` | Flutter 通道、文件选择和 MediaStore 保存 |
| `NativeImage.kt` | EXIF 方向及原始图像区域读取 |
| `PatchPipeline.kt` / `StreamingPng.kt` | 重叠 patch、行缓存、拼接和 PNG 写入 |
| `QnnRunner.kt` | 模型包校验与两种后端的调用入口 |
| `NativeQnn.kt` / `NativeGpu.kt` | JNI 接口和会话复用 |
| `vosr_jni.cpp` | QNN HTP context、性能投票及事件采集 |
| `ncnn_jni.cpp` | ncnn Vulkan 网络及 CPU 回退算子计时 |

Kotlin 文件位于 `android/app/src/main/kotlin/ai/vosr/vosr_restore/`，C++ 文件位于 `android/app/src/main/cpp/`。

输入保持原始分辨率。2× / 4× 输出按 512×512 块、448 步长处理；高斯权重融合重叠区域，缓存当前和上一行的输出块，逐行写出 PNG。模型子图间通过内存传递张量，磁盘行缓存用于输出拼接。

NPU 使用 12 个 QNN context；GPU 使用 12 个 ncnn Net。切换后端会释放上一组会话。GPU 子图之间目前通过 CPU Mat 传递数据，尚未保持为 VkMat；其中 134 个 MatMul 和 85 个 Reduction 走 CPU。精度配置和性能见 [阶段 10](../docs/10-performance-profile.md)。

模型导入按后端检查 manifest、文件名、大小限制和 SHA-256，校验完成后替换旧目录。手机上的 `models`、`models-gpu` 目录由应用创建；不要用 adb shell 提前创建，以免属主不匹配。

从仓库根目录执行本地检查：

```powershell
. .\scripts\env.ps1
Push-Location app
flutter analyze
flutter test
Pop-Location
```

`MainActivity` 提供仅在 debug 构建启用的 intent 测试入口，供 `scripts/test-mobile-backends.py` 和 `scripts/profile-mobile.py` 使用。正常界面默认关闭 QNN 详细事件和 CPU 逐层计时。测试通过文件与日志验收，不依赖截屏。

当前交付使用开发签名。发布版需要单独配置签名；不要提交签名文件、`key.properties` 或 `local.properties`。
