# 阶段 5：小米 13 应用内真机验收

完成日期：2026-11-14（Asia/Shanghai）。新版原始分辨率修订详见阶段 6。

## 通过的检查

- Flutter 静态分析及 Android APK 构建通过。
- APK 签名及打包的官方 QNN 原生库校验通过。
- 165 个官方模型文件通过 LFS SHA-256 或 Git blob SHA-1 校验。
- 完整 VOSR 在 shell HTP 执行的转换精度为 PSNR 60.1458 dB，详见阶段 3。
- 在普通应用 UID 下完整调用 12 个真实 HTP 子图，输出图像及逐图耗时已取回。
- 初版单块：512×512，12.522 秒；初版两块：512×768，33.702 秒。证据在 `artifacts/app-test/single` 和 `rectangle`。
- 后续原始分辨率版本的两项验收均通过：横向/纵向各 3 块、原始字节与方向保留、正确尺寸、流式 PNG，以及实际 MediaStore 相册文件的哈希一致性。详见阶段 6 和 `artifacts/app-test/native-resolution/acceptance.json`。

## 修复的问题

1. 用户开启 USB 安装后，流式 ADB 安装仍被小米拒绝；非流式安装成功。安装脚本已使用 `--no-streaming`。
2. ADB 创建的模型目录归 shell 所有，应用无法遍历。已改由应用自身创建目录，安装脚本等待该目录出现后再复制模型。已有模型已迁入正确目录，不需重新下载。
3. 用户指出整图预缩放不合适，已移除并实现原始区域解码与流式 patch 拼接，详见阶段 6。

## 内存数据边界

初版单块采样记录见 `evidence/app-memory-samples.csv`：主进程与 QNN 子进程合计采样 RSS 最大约 505.81 MiB。RSS 可能重复计算共享页，不含 DSP 内存，不能当作完整物理峰值；该值也不代表后续原始分辨率版本的任意大图内存。

## 验收方式与边界

按用户要求停止截屏，本次后续验收仅通过日志、文件、哈希、PNG 解码及 MediaStore 记录；本任务已生成的本地界面截图已清理。

验收入口复用正常图片导入与保存方法，实际在应用权限下执行。没有对系统文件选择器和滑动控件进行最终人工点击或视觉回归，不声称完成这类测试。

当前交付为开发签名 debug APK，模型离线运行，未进行商店发布。界面预览可缩小显示，模型输入和输出保留实际原始尺寸乘倍率。

## 交付

- `deliverables/vosr-restore-debug.apk`
- `deliverables/vosr-0.5b-htp-v73-models.zip`
- `deliverables/使用说明.md`
- `deliverables/SHA256.json`

安装与构建入口：`scripts/install-android.ps1`、`scripts/build-android.ps1`。原始分辨率验收入口：`scripts/test-native-resolution.py`。
