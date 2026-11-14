# 阶段 3：模型转换与小米 13 NPU 验证

完成日期：2026-11-14（Asia/Shanghai）。本阶段已完成完整模型的真实 HTP 执行，不含应用进程内集成的验收。

## 转换产物

将官方 VOSR 0.5B 单步模型拆为 12 个固定形状子图：VAE encoder、DINOv2 前 9 层特征提取、DiT prepare、7 组各 4 个 DiT block、DiT finish、轻量 decoder。

固定处理块为 512×512，latent 为 1×4×64×64，DiT token 为 1×1024×1024；DINO 输入为 1×3×448×448。固定单步 t=1 / r=0 的时间条件作为常量折叠。

每个子图先导出 ONNX opset 17、执行 ONNX checker、对照原始 PyTorch 数值，再通过 Qualcomm `qairt-converter` 转为 FP16 DLC，使用 `qnn-context-binary-generator` 编译成 HTP v73 context。

- ONNX 及精度记录：`artifacts/onnx/`。
- DLC、HTP context、每图编译配置：`artifacts/qnn/`。
- 编译时指定 HTP v73、soc_model=43、VTCM 8 MB；源和目标 IO 均显式保留 NCHW / NTF，IO 数据为 FP32，内部计算为 FP16。
- 实现与复现入口：`conversion/pipeline.py`、`conversion/export.py`、`conversion/build_qnn.py`。

## 真机执行结果

设备为此前连接并核实的 Xiaomi 13 / SM8550 / Android 13。使用官方 QAIRT 2.44.0 的 `qnn-net-run`，明确指定 `libQnnHtp.so`，加载每个真实 VOSR context。没有提供 CPU 后端或 CPU 回退路径。

全部 12 个子图成功执行，输出尺寸、有限数值检查和精度检查通过。完整流水线使用 NPU 前一子图的真实输出传给下一子图，未以桌面结果替代中间计算。

最终 512×512 图像相对完整 FP32 ONNX Runtime 流水线：

| 指标 | 实测值 |
| --- | --- |
| 像素 PSNR | 60.1458 dB |
| RGB tensor RMSE（[-1,1]） | 0.00196671 |
| RGB tensor NRMSE | 0.00706618 |
| 最大绝对误差 | 0.14281464 |

该对比用于验证转换精度，不是对图像复原质量或原图真实性的评分。

## 可复核证据

- 最终 NPU 图像：`artifacts/device512/restored-npu.png`。
- 完整逐图精度报告：`artifacts/device512/validation.json`。
- 各子图标准输出、raw tensor、QNN 原始 profiling log、解析后的 CSV：`artifacts/device512/`。
- Profiling 中存在 HVX 线程、HMX 电源及 accelerator execution/cycles 事件，和明确使用 HTP backend 的执行命令互相印证。
- 性能汇总：`artifacts/device512/performance.json`。当前为单次参考执行，不是稳定温度下多轮平均，也不代表 Flutter 应用端到端耗时。
- 该次 12 个子图的 QNN NetRun 执行时间之和为 3832.52 ms；不含逐图进程启动、context 加载、应用图像预处理及合成等开销。
- 原始参考输入、固定噪声与 FP32 结果：`artifacts/reference/`。

早期 256 解码器与 512 完整链路的测试目录曾复用。已纠正最终 decoder profiling log 编号，并修改验证脚本使用每次唯一的输出目录，避免后续混用记录。

## 复现

```powershell
. E:\work\ai\vosr-mobile\scripts\qnn-env.ps1
E:\work\ai\vosr-mobile\.venv-qnn\Scripts\python.exe E:\work\ai\vosr-mobile\conversion\build_qnn.py --workers 2
E:\work\ai\vosr-mobile\.venv\Scripts\python.exe E:\work\ai\vosr-mobile\conversion\validate_device.py
```

`validate_device.py` 使用序列号 `cf6932c0`，并假定官方运行库已部署于 `/data/local/tmp/vosr/qnn`。运行库和已编译 context 必须使用一致 SDK 版本。

## 下一阶段

将相同 context 与 HTP 运行库接入 Flutter 应用，并验证普通 Android 应用身份下的 FastRPC / HTP 可访问性。ADB shell 下运行成功不能代替应用沙箱验证。
