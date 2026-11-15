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

这个结果不证明系统相机独立拍摄时也不写该字段。外部拍照模式与独立拍照可能走不同的保存流程，需要使用同一台手机、同样的 10× 设置，比较两条路径返回的原片。不能根据现有结果直接认定系统相机永远不提供变焦信息，也不能用 23 mm 焦距猜测用户实际使用的总变焦。

## 诊断入口

```powershell
# 仅导出指定拍照文件的白名单元数据
adb shell am instrument -w -e captureFile capture_<文件名>.jpg ai.vosr.vosr_restore/.CameraMetadataChecks
# 独立系统相机原片通过应用“选择照片”导入后，检查应用保留的原始字节副本
adb shell am instrument -w -e importFile input_<时间戳>.img ai.vosr.vosr_restore/.CameraMetadataChecks
```

两个参数只允许应用缓存里对应格式的文件名，不能指定任意路径。摘要同时保留 AndroidX 和系统解析结果。完整摘要留在本地 `artifacts/camera-tests/` 及 `evidence/camera-capture-metadata.log`，不提交原图或照片哈希。

另外修正了错误提示：EXIF 读取失败、变焦字段缺失、字段值无效和低于门槛分开处理，不再把任何解析异常都显示成“系统相机没有提供 EXIF”。严格的 ≥10× 校验没有放宽。

诊断版本已编译。独立相机原片的对照尚未完成，因此没有把外部调用模式、系统相机保存策略或其它原因中的任何一项写成已确认根因。
