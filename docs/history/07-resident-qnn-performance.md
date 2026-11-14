# 阶段 7：JNI 常驻 QNN 会话与性能优化

完成日期：2026-11-14（Asia/Shanghai）。新版已构建并安装到连接的小米 13；全过程仅检查日志和文件，未截屏或读取屏幕。

## 改了什么

旧版每个 patch 启动 12 次原生执行进程，每次重新加载对应 HTP context，子图之间通过浮点文件传递张量。3 个 patch 会启动 36 次进程，阶段 6 的两个样例分别需要 33.103 秒、33.769 秒。

现在由 C++ JNI 在应用进程内直接调用官方 QNN HTP API：首次加载 12 个 context，后续 patch 和下一张照片复用。模型通过 mmap 读取，System context 元数据和模型映射在会话生命周期内保留。更换模型包时释放会话；进程退出后下次重新加载。

子图的输入、输出绑定到可复用的原生浮点缓冲区；DiT tokens 使用两个缓冲区交替读写，避免输入输出原地别名。Kotlin 只传入当前 patch 的 RGB、DINO 条件和噪声，接收最终解码像素。取消操作在子图间检查，原生句柄通过共享所有权管理，避免释放与调用竞争。

推理期间不再启动子进程，不再把子图中间张量写入磁盘。这不等于 NPU 零拷贝：JNI 数组和 QNN 原始缓冲区接口仍可能复制数据。最终解码 patch 的行缓存文件仍保留，用于原始分辨率流式拼接；它们不是子图间传输文件。

保留阶段 6 的原始输入、EXIF 区域解码、512 输出 patch、448 步长、64 重叠、高斯融合及逐行 PNG 写出。模型、精度和独立模型 ZIP 均未改变。

## 真机结果

设备：小米 13，SM8550 / HTP v73，Android 13；普通应用 UID。两个固定样例均为 2×、3 个 patch，顺序在同一应用进程内处理；第二个样例不重启应用。

| 样例 | 旧版处理 | 新版处理流水线 | 新版总耗时 | 模型校验 / 会话初始化 | 流水线加速 |
| --- | --- | --- | --- | --- | --- |
| PNG 横向，首次处理 | 33.103 秒 | 7.740 秒 | 11.448 秒 | 1.417 / 2.279 秒 | 4.28 倍 |
| JPEG，EXIF=6，复用会话 | 33.769 秒 | 7.061 秒 | 7.064 秒 | 0 / 0 秒 | 4.78 倍 |

旧版 `elapsedMs` 从 patch 流水线开始计时，不含此前模型校验；新版 `pipelineMs` 与之比较。新版 `elapsedMs` 则包含本次完整模型校验、会话准备和处理。因此首次处理的用户可见耗时从约 33 秒降至 11.448 秒，后续本例约 7.064 秒。相册保存由完成后的独立操作验证，不计入这些耗时。每个 patch 的完整 JNI 推理约 1.69–1.71 秒，已包含 12 个子图执行及 JNI 输入输出。

两组验收均通过：

- 与阶段 6 留存输出逐像素完全一致，最大像素差为 0。
- PNG 输入逻辑尺寸 514×129，输出 1028×258；EXIF=6 的 JPEG 逻辑尺寸 129×514，输出 258×1028。
- `inputDownscaled=false`、`streamingOutput=true`、3 个 patch，严格按倍率输出。
- 首张完成后创建 context 数为 12、累计处理 3 个 patch；第二张完成后 context 仍为 12、累计 6 个 patch，`residentEngineReused=true`。
- 原生执行路径子进程数为 0、子图中间张量文件字节数为 0；实际 backend 为 QNN HTP，无 CPU 回退。
- MediaStore 条目 `is_pending=0`，相册副本与输出 PNG 的 SHA-256 一致。

第二张结束后的 `dumpsys meminfo`：应用 PSS 1,523,563 KB（约 1.45 GiB），RSS 1,613,316 KB（约 1.54 GiB），其中 Other mmap 约 1.24 GiB，主要包含常驻模型映射。该读数是处理结束后的进程内存，不是峰值，也不覆盖所有 DSP 私有分配。会话常驻以内存换取加载时间；模型映射中干净页可由系统回收，应用进程被回收后需重新初始化。

上述是每例一次实测，不是持续负载平均值。大图耗时仍随 patch 数增长；本次优化移除了重复初始化与文件传输，未减少模型计算量。

## 文件与复现

- `app/android/app/src/main/cpp/vosr_jni.cpp`：QNN 动态加载、context、张量绑定、执行、取消及计时。
- `app/android/app/src/main/kotlin/ai/vosr/vosr_restore/NativeQnn.kt`：JNI 接口和跨照片会话缓存。
- 同目录 `QnnRunner.kt`、`PatchPipeline.kt`：数组推理接口与完整计时；`MainActivity.kt` 支持同进程 debug 连续验收。
- `scripts/build-native.ps1`：使用已安装的 NDK r28c 和 QAIRT 2.44 头文件编译 ARM64 JNI 库；`build-android.ps1` 自动先构建该库。
- APK 打包 `libvosr_jni.so` 和原有 QNN 库，移除旧的 `libvosr_runner.so`。
- `scripts/benchmark-resident.py`：首次加载、会话复用、输出像素、尺寸、EXIF、相册副本自动验证，不涉及屏幕。

```powershell
E:\work\ai\vosr-mobile\scripts\build-android.ps1
E:\work\ai\tools\android-sdk\platform-tools\adb.exe -s cf6932c0 install --no-streaming -r E:\work\ai\vosr-mobile\deliverables\vosr-restore-debug.apk
E:\work\ai\vosr-mobile\.venv\Scripts\python.exe E:\work\ai\vosr-mobile\scripts\benchmark-resident.py
```

构建成功记录：`evidence/resident-jni-build.log`。真机记录：`evidence/resident-jni-device-test.log`；结构化结果、逐子图计时、输出和相册副本位于 `artifacts/app-test/resident-jni/`，汇总为 `acceptance.json`，内存记录为 `resident-memory.txt`。基准旧 APK 保存在 `artifacts/baselines/file-runner-debug.apk`。

交付 APK 为 `deliverables/vosr-restore-debug.apk`，校验和见 `deliverables/SHA256.json`。独立模型包可继续使用，无需重新下载或转换。

最终 APK 的 v2 签名验证通过；打包的 5 个原生库与构建输入逐字节一致，旧 runner 不在 APK 内，见 `evidence/apk-native-verification.json`。新版已覆盖安装，并同步到手机下载目录 `VOSR-Restore-1.0-debug.apk`；返回现有应用实例时未重启进程。
