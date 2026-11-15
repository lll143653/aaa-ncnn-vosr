# 阶段 13：拍照返回文件的 EXIF 排查

日期：2026-11-15。

排查问题：小米 13 在 VOSR 中拍照后提示缺少变焦 EXIF，是否由相机调用或文件处理造成。

已检查的调用为 `ACTION_IMAGE_CAPTURE`，通过 FileProvider 的 `content://` URI 放入 `MediaStore.EXTRA_OUTPUT`，同时授予临时读写权限。返回后直接读取预先指定的文件；没有读取 `data.extras["data"]` 缩略图，也没有在变焦校验前解码、压缩或保存照片。该方式与 [Android 完整照片调用示例](https://developer.android.com/media/camera/camera-deprecated/photobasics) 一致。

为避免把私人照片传到电脑，新增 debug instrumentation `CameraMetadataChecks`，在手机端读取白名单字段，只输出尺寸、哈希和选定 EXIF。原始 JPEG 不导出；摘要不包含画面、GPS 或拍摄时间。

## 从 VOSR 调起相机的返回文件

| 项目 | 手机端读取结果 |
| --- | --- |
| 文件类型 | JPEG |
| 文件大小 | 2,088,805 字节 |
| 图像尺寸 | 3072×4096 |
| Make / Model | Xiaomi / 2211133C |
| EXIF 版本 | 0220 |
| FocalLength | 5430/1000 mm |
| FocalLengthIn35mmFilm | 23 mm |
| DigitalZoomRatio | 字段缺失 |

AndroidX ExifInterface 1.4.1 和 Android 系统 ExifInterface 对这些字段的读取结果一致。可以确认这张返回图不是预览缩略图，也不是完全没有 EXIF；缺少的是当前校验使用的 `DigitalZoomRatio`。

## 独立系统相机对照

用户随后从桌面直接打开小米相机，按要求设置 10×，拍照后在 VOSR 中选择原片。选图入口将文件字节直接复制到缓存，校验前不缩放或重编码。

为避免打断手机上已有的复原任务，本轮没有安装新 APK 或启动 instrumentation，而是在本应用 UID 内运行独立的只读 JPEG/EXIF 探针。它只解析白名单字段，不解码照片画面，也不输出原始图像数据。探针先对照已由 AndroidX 与系统读取器检查过的返回文件，确认尺寸、厂家、型号、焦距和变焦字段一致。

另用已知包含 `DigitalZoomRatio=10/1` 的合成样例作正对照，探针成功读出 10/1，排除了探针对该字段始终漏读的情况。

| 字段 | VOSR 调起系统相机 | 桌面独立打开相机后选原片 |
| --- | --- | --- |
| 文件大小 | 2,088,805 字节 | 3,912,526 字节 |
| 图像尺寸 | 3072×4096 | 3072×4096 |
| Make / Model | Xiaomi / 2211133C | Xiaomi / 2211133C |
| EXIF 版本 | 0220 | 0220 |
| FocalLength | 5430/1000 mm | 5430/1000 mm |
| FocalLengthIn35mmFilm | 23 mm | 23 mm |
| DigitalZoomRatio | 缺失 | 缺失 |

相机 Intent 在本机解析到 `com.android.camera/.OneShotCamera`。两条路径都保留了其它 EXIF，也都没有标准变焦字段，因此本次对照不支持“VOSR 的外部调用独自弄丢了该字段”的判断。现有证据只能确认：在这台手机当前测试的拍照和保存环境中，不能依赖 `DigitalZoomRatio` 来严格验证界面上的 10×。不能把这个结论推广到所有小米机型、固件或拍照模式，也不能用 23 mm 焦距猜测实际总变焦。

严格门槛保持不变，两条样例都不能据此获得放行。真正需要调整的是取得可验证变焦值的方案，而不是把缺失值填成 10。

## 诊断入口

```powershell
# 仅导出指定拍照文件的白名单元数据
adb shell am instrument -w -e captureFile capture_<文件名>.jpg ai.vosr.vosr_restore/.CameraMetadataChecks
# 独立系统相机原片通过应用“选择照片”导入后，检查应用保留的原始字节副本
adb shell am instrument -w -e importFile input_<时间戳>.img ai.vosr.vosr_restore/.CameraMetadataChecks
```

两个参数只允许应用缓存里对应格式的文件名，不能指定任意路径。摘要同时保留 AndroidX 和系统解析结果。完整摘要留在本地 `artifacts/camera-tests/` 及 `evidence/camera-capture-metadata.log`，不提交原图或照片哈希。

Instrumentation 入口会影响目标应用进程，应在没有复原任务运行时使用。当前这次独立拍照对照的只读探针日志为 `evidence/camera-independent-comparison.log`，摘要保存在 `artifacts/camera-diagnostics/comparison.json`；这些文件仅留在本地。

另外修正了错误提示：EXIF 读取失败、变焦字段缺失、字段值无效和低于门槛分开处理，不再把任何解析异常都显示成“系统相机没有提供 EXIF”。严格的 ≥10× 校验没有放宽。

诊断版本已编译，独立相机原片对照已完成。为了不打断当前应用任务，本轮没有再次重装应用或启动模型推理。
