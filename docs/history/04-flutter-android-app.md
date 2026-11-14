# 阶段 4：Flutter 与 Android 原生应用

完成日期：2026-11-14（Asia/Shanghai）。已完成应用实现和 APK 构建；普通应用身份下的真机验收单独记录于阶段 5。

更新：用户要求的原始分辨率 patch 修订已完成，最新输入与拼接行为以 `06-native-resolution-patches.md` 为准，已替换初版预缩放策略。

性能更新：阶段 7 已用进程内 JNI 常驻会话替换下面记录的初版原生执行程序方案；最新架构及实测见 `07-resident-qnn-performance.md`。

## 实现内容

- Flutter 中文界面：图像选择、2× / 4× 放大、处理阶段与进度、原图/结果滑动对比、保存到相册、ZIP 模型包导入及校验。
- Android Storage Access Framework 选择图片和模型包；不申请整个相册的读取权限。
- 最新版本保留原始编码字节，通过区域解码处理 EXIF 和 patch，不预缩小整图，不再限制为 2048 输出。
- Kotlin 后端提供双三次插值、DINO ImageNet 归一化、固定种子的高斯噪声、512 像素分块、64 像素重叠及高斯混合。
- `MethodChannel` 提供命令，`EventChannel` 提供进度。耗时计算与文件操作在工作线程执行，不阻塞 Flutter 主线程。
- 通过随 APK 打包的官方 ARM64 QNN 原生执行程序调用 HTP；每次只加载一个 context，完成后释放该原生进程。该选择便于控制常驻内存，但包含进程创建和文件 IO 开销。
- 显式使用 HTP backend，无 CPU 自动回退；执行失败会保留诊断日志并向界面返回错误。
- 模型包检查图数量、设备后端、真机验证标记、文件名、尺寸及 SHA-256；导入支持临时目录校验后替换，限制解压总量并阻止 ZIP 路径穿越。
- 结果由 MediaStore 写入 `Pictures/VOSR`，使用 IS_PENDING，保存失败时删除未完成条目。

## 目录与接口

| 文件 | 作用 |
| --- | --- |
| app/lib/main.dart | Flutter 页面及原图/结果对比 |
| app/android/app/src/main/kotlin/ai/vosr/vosr_restore/MainActivity.kt | 平台通道、文件选择、相册保存、debug 验收入口 |
| app/android/app/src/main/kotlin/ai/vosr/vosr_restore/QnnRunner.kt | 模型包管理、图像流水线、HTP 调度 |
| app/android/app/src/main/jniLibs/arm64-v8a | 官方 QNN / HTP v73 库及原生执行程序 |
| scripts/build-android.ps1 | 一键构建并整理 APK |
| scripts/install-android.ps1 | ADB 安装、模型部署、可选应用内复原测试 |

## 构建设置

Flutter 3.50.0 / Dart 3.14.0，Material UI 1.1.0，Android Gradle Plugin 9.3.1，Gradle 9.5.0，Kotlin 2.4.20，JDK 17，Android SDK 36，NDK r28c（28.2.13676358）。最低 Android API 29，仅打包 arm64-v8a。

安装 QNN 原生库时要求实际解压到 nativeLibraryDir；声明厂商 `libcdsprpc.so`。QNN 库与 Hexagon skeleton 保留原始调试符号，避免构建时重写厂商二进制。

首次构建发现 Gradle 网络下载和 Kotlin 守护进程握手问题。已校验并使用本地 Gradle 分发包，配置现有本机代理，改用进程内 Kotlin 编译。新版 Android CLI 的 NDK 包名需使用斜杠，因此显式安装了完整 NDK r28c。

## 校验与产物

Flutter 静态分析通过；Android `assembleDebug` 构建成功。原始构建日志位于 `evidence/gradle-build.log`，最终交付构建记录位于 `evidence/final-android-build.log`。

交付 APK 为开发签名 debug 版本，适合当前手机的开发验证；没有进行商店发布或生产签名配置。

- APK：`deliverables/vosr-restore-debug.apk`。
- 独立离线模型包：`deliverables/vosr-0.5b-htp-v73-models.zip`，约 1.26 GB。
- 组件和许可：`THIRD_PARTY_NOTICES.md` 及 `deliverables/` 中保留的原始许可。

## 复现

```powershell
E:\work\ai\vosr-mobile\scripts\build-android.ps1
E:\work\ai\vosr-mobile\scripts\install-android.ps1 -Serial cf6932c0 -RunSmokeTest
```

所有图像复原计算都在手机执行。桌面上的模型转换、推理参考和性能对比工具不参与应用运行。
