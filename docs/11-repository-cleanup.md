# 阶段 11：仓库整理

日期：2026-11-14。基线为 `26e8a46`，开发分支为 `feat/mobile-npu-gpu`。

这次把 Android 双后端、模型转换、原始分辨率 patch、真机测试及 Profile 工具一起纳入版本管理。桌面 1.4B 的源码和模型参数保留，原 README 的桌面用法移到 `docs/desktop.md`。

README 现在只介绍项目入口、两种模型的区别、手机功能和实测性能。`docs/android.md` 负责安装、工具链、转换和测试；`app/README.md` 说明代码结构。阶段 8–10 保留当时的转换、兼容性测试和性能记录，早期独立应用记录仍在 `docs/history/`。

提交前修正了几处仓库问题：

- 旧 `*_ncnn.py` 规则会忽略手写的 `export_ncnn.py`、`validate_ncnn.py`，现在明确纳入版本管理。
- 补入 Gradle Wrapper 和启动脚本，将本机 ZIP 路径换成官方发行地址，保留 SHA-256 校验。
- 移除仓库内固定的个人代理；设备脚本改为接受显式序列号、`ANDROID_SERIAL` 或单设备自动选择，Profile 动态读取应用 UID。
- 排除 JVM heap dump、构建产物、签名文件、APK、模型包、SDK 和大体积 trace。
- 清除 Flutter 模板介绍与无关注释，修正遗留转换配置中的旧项目路径。
- 保存一份精简的 JSON 测量摘要，方便在仓库中查看逐次性能数据；完整采样仍在本地。

本轮检查包括 Flutter 静态分析和组件测试、Gradle 官方发行包下载与校验、Android 编译、Python/PowerShell 语法以及已有 25 组 Profile 结果的验收。没有重新转换模型或修改推理算术；真机数值结果与性能条件见阶段 9、10。

提交包含源码、构建脚本、测试图片、文档、模型清单和许可文件。约 733 MiB 的 JVM 崩溃转储留在本地并被忽略；APK、模型权重和运行库也不进入 Git。当前本地安装包的哈希见 `deliverables/SHA256.json`。
