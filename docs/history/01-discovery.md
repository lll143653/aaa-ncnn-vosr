# 阶段 1：模型来源、设备与部署方案

完成日期：2026-11-14（Asia/Shanghai）。本阶段完成的是调查和方案确认，不代表模型已经在手机上运行。

## 已完成

1. 通过 ADB 读取手机属性并保存至 `evidence/device-inventory.txt`：Xiaomi 13 / 2211133C、SM8550（kalama）、Android 13 / API 33、arm64。
2. 手机内存总量为 11,430,784 KiB，查询时可用约 6 GB；数据分区可用约 97 GB。设备存在 `/vendor/lib64/libcdsprpc.so`，但这不能单独证明第三方应用能成功调用 NPU。
3. 确认官方项目为 `cswry/VOSR`，权重为 `CSWRY/VOSR`。保存官方文件清单、模型卡和版本信息于 `evidence/`。
4. 下载源码至 `third_party/VOSR-main`。原有 `E:\work\ai` 内项目不作改动。
5. 建立工作目录隔离脚本、带断点续传和大模型 SHA-256 校验的下载脚本、阶段状态表。

## 模型选择及执行链路

先部署官方 `VOSR_0.5B_os` 单步模型。其权重为 1,948,337,520 字节 FP32，需要 SD2.1 VAE、DINOv2 ViT-B/14 和官方轻量解码器。所选配套文件总计约 2.68 GB。

源码中的实际链路：

1. 输入按目标倍数双三次插值，转换为 RGB / NCHW / [-1, 1]。
2. VAE 编码使用 posterior mode（确定性），乘 VAE scaling_factor。
3. DINOv2 输入 448×448，按 ImageNet 均值方差归一化，取第 8 层（从 0 开始）特征。
4. 28 层、隐藏维度 1024 的 DiT，单步从 t=1 到 r=0，输出速度场，`z_out = noise - velocity`。
5. 除以 VAE scaling_factor，经官方 LightDecoder 解码；可应用官方颜色校正。

第一条验证链路使用固定小尺寸输出和相同噪声，先确保导出数值一致。更大图像采用有重叠区域的分块策略，需另行验证接缝和内存；不能把小尺寸成功当成任意分辨率已通过。

## NPU 与应用路线

- 使用 Qualcomm QAIRT / QNN 的 HTP 后端适配 SM8550，先用小型计算图验证 NPU 可访问性，再部署真实 VOSR 子图。
- 模型按 VAE encoder、DINO、DiT 子图、decoder 分拆，避免整图常驻及单次编译内存过高。
- 优先评估 HTP FP16，随后在有质量对比数据时评估量化。仅成功导出 ONNX 不能视为 NPU 转换成功。
- Flutter 提供选择图像、处理进度、结果对比和保存界面，Android/C++ 层执行推理；推理不依赖云服务。
- 真机验收要求：实际恢复输出、NPU 后端证据、耗时、峰值内存、异常信息。不得将静默 CPU 回退展示为 NPU 成功。

## 工具与环境

电脑可查询到约 31 GiB 物理内存。已有 Python 环境位于其他项目，故另行安装项目内 Python 3.11。现有 NVIDIA 查询和 PyTorch 环境未提供可用 CUDA 证据，参考推理先按 CPU 准备。

新工具目录：`E:\work\ai\tools`；下载：`E:\work\ai\downloads`；缓存：`E:\work\ai\cache`；模型：`E:\work\ai\models\vosr`。不更改系统 PATH，不覆盖已有 Python 包。

GitHub 直连存在超时；源码压缩包通过电脑现有的 `127.0.0.1:7897` 代理下载。依赖清单和日志保存于 `evidence/`。

## 官方依据

- 项目与推理源码：https://github.com/cswry/VOSR
- 模型与各组件许可：https://huggingface.co/CSWRY/VOSR
- QNN 官方文档：https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
- Qualcomm SDK 官方开源下载示例：https://github.com/qualcomm/qai-appbuilder/blob/main/BUILD.md
- Flutter SDK：https://docs.flutter.dev/install/archive

## 后续验收

工具链下载和模型 SHA-256 校验尚未全部完成。下一阶段需要提供成功的参考推理图像及环境版本清单；第三阶段才记录转换与 NPU 的成功或失败证据。
