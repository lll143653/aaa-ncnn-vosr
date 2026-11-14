# 阶段 6：原始分辨率 patch 复原与流式拼接

完成日期：2026-11-14（Asia/Shanghai）。落实用户修正：不先缩小整张输入图，保留原始像素，切 patch 推理后拼接。

后续更新：本报告的 33 秒耗时为文件/子进程版基准。阶段 7 保留本阶段输入和拼接行为，改用 JNI 常驻 QNN，会话复用样例降至 7.064 秒；见 `07-resident-qnn-performance.md`。

## 修改

- 图片导入直接复制原始编码字节，不缩到长边 512，不重新压缩；保留 EXIF。
- BitmapRegionDecoder 以 inSampleSize=1 读取原始小区域，按 EXIF 映射旋转/镜像坐标。
- 模型输出 patch 为 512×512、步长 448、重叠 64。2× 时读取约 256×256 原始区域，4× 时约 128×128，另有插值 halo。
- 仅对当前 patch 按 SR 倍率插值形成模型条件；DINO 分支按模型要求使用 448×448。这不改变整张原图的尺寸或采样密度。
- 重叠区域高斯加权融合，只缓存当前与上一行 patch 的临时文件；逐扫描行合成 PNG，不创建完整输出 Bitmap/float tensor。
- 按全局 latent 坐标和种子生成噪声，重叠位置噪声一致，不分配整幅噪声数组。
- 删除 2048 输出限制，改为实际磁盘空间检查。界面显示原始/输出尺寸及 patch 数，增加取消处理。
- Flutter cacheWidth=1024 仅用于界面预览；模型读取未经缩小的原始文件。

实现：`NativeImage.kt`、`PatchPipeline.kt`、`StreamingPng.kt`、`QnnRunner.kt`、`MainActivity.kt`，位于 `app/android/app/src/main/kotlin/ai/vosr/vosr_restore/`。

## 真机验收（不截屏）

新版 APK 在同一小米 13 的普通应用 UID 下，调用与正常流程相同的导入、HTP 复原和 MediaStore 保存方法。

| 样例 | 原始逻辑尺寸 | 输出尺寸 / 倍率 | patch 数 | 处理耗时 |
| --- | --- | --- | --- | --- |
| PNG 横向 | 514×129 | 1028×258 / 2× | 3 | 33.103 秒 |
| JPEG，EXIF=6 | 129×514（编码尺寸 514×129） | 258×1028 / 2× | 3 | 33.769 秒 |

两组全部通过：导入 SHA-256 与原始样例一致；`inputDownscaled=false`、`streamingOutput=true`；严格倍率输出；HTP FP16 完整执行；浮点像素有限、无未覆盖扫描行；PNG CRC/格式/尺寸与非空内容通过；MediaStore 返回真实 URI 且 `is_pending=0`；相册副本与输出 PNG 的 SHA-256 完全一致。

证据：`artifacts/app-test/native-resolution/acceptance.json`，各样例目录中的结果 JSON、逐子图耗时、输出及相册副本、MediaStore 元数据。日志：`evidence/native-resolution-device-test.log`。

Flutter 静态分析和 Android 构建通过，见 `evidence/native-resolution-analyze.log` 和 `evidence/native-resolution-build.log`。

复现：运行 `scripts/build-android.ps1`；安装后运行 `.venv/Scripts/python.exe scripts/test-native-resolution.py`。

大图耗时随 patch 数增加；上述为单次样例，不是任意大图的性能保证。模型接口未改变，无需重新下载或转换模型。

按用户要求，此次修正及验收仅使用日志、输出文件、尺寸、哈希与 MediaStore 记录；不截屏、不读取屏幕。
