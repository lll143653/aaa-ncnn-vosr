# 阶段状态

日期：2026-11-15。

| 阶段 | 状态 | 文档 |
| --- | --- | --- |
| 1–5 模型、工具链、转换、NPU 和原应用 | 已完成，历史记录归档 | history/01-discovery.md 至 history/05-device-acceptance.md |
| 6 原始分辨率 patch | 已完成，融合版继续保留 | history/06-native-resolution-patches.md |
| 7 JNI 常驻 QNN | 已完成，融合版补充执行期性能投票 | history/07-resident-qnn-performance.md |
| 8 仓库融合与 ncnn 转换 | 已完成 | 08-repository-integration.md |
| 9 双后端真机验收 | 已完成，连续切换、尺寸、EXIF、数值和保存均通过 | 09-mobile-backend-acceptance.md |
| 10 应用内性能 Profile | 已完成，前台/休眠分组、12 子图硬件事件、全部 CPU 回退算子和数值回归 | 10-performance-profile.md |
| 11 仓库整理 | 已完成，构建文件、文档入口和提交范围整理 | 11-repository-cleanup.md |
| 12 系统相机拍照复原 | 代码、原生数值检查和 NPU 样例通过；本机实拍因 EXIF 缺失被严格拒绝 | 12-camera-restoration.md |
| 13 拍照元数据排查 | 对照完成；两条相机路径均保留 EXIF，均缺少标准变焦字段 | 13-camera-metadata-diagnosis.md |
| 14 输入块尺度修正 | 已完成；832×624 实际 4 块，NPU 整图保存、相机 EXIF 分支和 GPU 单次 JNI 数值检查通过 | 14-input-patch-512.md |
| 15 SM8850 新机适配 | 已完成；v81 NPU 四块整图与保存通过，Adreno 840 完整 JNI 数值通过，VAE 使用 GPU | 15-sm8850-adaptation.md |

当前用法见 [Android 文档](android.md)，本地安装包说明见 [使用说明](../deliverables/使用说明.md)。阶段 10 的 1.588 秒 NPU、38.55 秒 GPU 均为旧 512 输出块的历史测量，不能套用到新 1024 输出规格。开发分支为 `feat/mobile-npu-gpu`。
