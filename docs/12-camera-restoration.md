# 阶段 12：系统相机拍照复原

日期：2026-11-15。

新增“拍照并复原（≥10×）”入口。模型就绪后，按钮打开系统相机；拍摄完成并通过 EXIF 校验后，自动进行复原。拍照取消或校验失败不会进入模型推理。相册选图继续使用原有的原始分辨率分块流程。

## 变焦门槛

按严格规则检查原始照片的 EXIF `DigitalZoomRatio`：数值有效且不小于 10 才处理。缺失、0、无效数值及低于 10 均拒绝，没有手动放行选项。复原入口会再次检查原始文件，不接受界面传入的倍数替代 EXIF。

这个 EXIF 字段表示数字变焦，不一定等于系统相机界面显示的总变焦。应用没有根据焦距猜测光学变焦，也不使用厂商私有参数强制设置相机。若系统相机未保存该字段，或界面显示 10× 但 EXIF 记录不足 10，就会按严格规则拒绝。

本机实拍已返回应用，用户确认显示“没有 EXIF 参数”。因此这次小米系统相机返回的照片没有通过校验，未进入超分。当前已验证的是系统相机返回及拒绝分支；真实系统相机照片满足门槛后的自动复原，尚未在这台手机上验收。保持严格规则时，不会通过手动确认或补写 EXIF 放行。

后续[手机端元数据排查](13-camera-metadata-diagnosis.md)确认：原文件为 3072×4096 JPEG，其它 EXIF 正常存在，缺少的是 `DigitalZoomRatio`，不是所有 EXIF 都丢失。

使用标准 `ACTION_IMAGE_CAPTURE`，通过 FileProvider 接收完整照片文件，忽略返回 intent 中的预览缩略图。授权仅覆盖应用缓存里的拍照文件，返回或取消后撤销授权；不需要应用直接持有 CAMERA 权限。拍照返回 intent 没有 data URI 时，仍从预先指定的输出文件读取。相机引起 Activity 重建时，可从保存的拍照路径恢复待处理照片。

## 处理流程

1. 读取 EXIF 方向，得到拍照原图的逻辑宽、高。
2. 等比例缩放到最长边 **832**。大图缩小，小图也放大到 832；短边取最近整数，最小为 1。
3. 固定 **2×**，生成最长边 1664 的中间复原图。沿用 512×512 输出 patch，重叠改为 **96 个输出像素**，步长 416。重叠部分仍按高斯权重拼接。
4. 将中间复原图等比例重采样回拍照原图的逻辑尺寸。
5. 与原始 LR 做五层高斯金字塔融合，输出同原图尺寸的 PNG，可保存到相册。

例如 4000×3000 的拍照原图依次经过 832×624、1664×1248，最后输出 4000×3000。内部使用 2× 超分，最终文件不会变成 8000×6000。

金字塔使用 `[1,4,6,4,1]/16` 的可分离高斯核，逐层降采样两倍，再在同一网格上双线性展开。融合保留 SR 的细节与 LR 的低频颜色，等价于：

```text
结果 = SR + Expand(Gaussian5(LR) - Gaussian5(SR))
```

计算后裁到 RGB 的有效范围。原图按区域读取，每个融合块带 128 像素边缘，块原点对齐 32 像素网格，以覆盖五层金字塔的滤波范围。结果按行写 PNG，不在 Java 堆里同时保存两份全分辨率浮点金字塔。模型的处理尺寸固定，但最终融合仍需遍历原图全部像素，所以这部分耗时随原始尺寸增长。

## 代码和验证入口

| 文件 | 内容 |
| --- | --- |
| `app/lib/main.dart` | 拍照入口、自动复原和固定 2× 状态 |
| `MainActivity.kt` | 系统相机、文件授权、返回处理及恢复 |
| `CameraPhoto.kt` | 严格 EXIF 校验与 832 尺寸计算 |
| `CameraPipeline.kt` | 相机专用处理、原尺寸融合及临时文件清理 |
| `NativeImage.kt` / `photo_jni.cpp` | EXIF 区域重采样和金字塔融合 |

Kotlin 文件位于 `app/android/app/src/main/kotlin/ai/vosr/vosr_restore/`。新增 `libvosr_photo.so`，由原有构建脚本编译；QNN、ncnn 模型和推理算术没有变化。两个后端共用拍照流水线。

```powershell
.\scripts\build-android.ps1
.\scripts\install-android.ps1 -Serial <设备序列号> -SkipModels
# 手机端数值检查，不打开相机或截屏
E:\work\ai\tools\android-sdk\platform-tools\adb.exe shell am instrument -w ai.vosr.vosr_restore/.CameraChecks
# 合成 EXIF 输入的完整 NPU 回归，会保存测试图片到相册
.\.venv\Scripts\python.exe scripts\test-camera.py --serial <设备序列号>
```

`CameraChecks` 检查 832 的放大/缩小、10× 边界、8 种 EXIF 方向、金字塔同图恒等性、LR 低频颜色，以及整图与分块计算的一致性。该入口仅编入 debug APK。Flutter 测试检查有效拍照输入自动走固定 2× 的相机流程、EXIF 拒绝时不启动推理。

完整样例与结果保存在本地 `artifacts/camera-tests/`，构建及数值日志在 `evidence/camera-*.log`。合成 EXIF 用于验证程序分支，不能代表系统相机一定会写入该字段。

仓库保留[测量摘要](benchmarks/camera-2026-11-15.json)，包含各样例的输入、工作图、中间 SR 和最终输出尺寸。

本次小米 13 的自动检查结果：

| 输入 | 校验与处理结果 |
| --- | --- |
| 无变焦 EXIF、9× EXIF | 拒绝，未进入推理 |
| 100×50、10× | 工作图 832×416，8 块，输出 100×50 |
| 编码尺寸 1001×751、EXIF 方向 6、12× | 工作图 624×832，12 块，输出 751×1001 |
| 4000×3000、10× | 工作图 832×624，12 块，输出 4000×3000 |

以上有效样例均实际调用 NPU，并完成相册写入。最终版 4000×3000 样例总耗时 39.915 秒，其中原尺寸融合 7.105 秒；12 块原生推理合计约 19.5 秒。本轮没有固定频率或温度，不是稳定状态性能基准。两个小尺寸样例在重采样优化前完成，表中只记录它们的功能结果。

原生检查另外确认：8 种方向与既有读取器一致；分块与整图金字塔结果最大差为 0；相同输入保持像素，LR 低频颜色和 SR 高频纹理均保留。Flutter 静态分析及 3 个组件测试通过。GPU 共用新增流水线，但本轮新增拍照路径的完整模型验收使用 NPU。

系统相机文件输出方式参照 [Android 拍照文档](https://developer.android.com/media/camera/camera-deprecated/photobasics)；EXIF 字段说明见 [AndroidX ExifInterface](https://developer.android.com/reference/androidx/exifinterface/media/ExifInterface)。
