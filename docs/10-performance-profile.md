# 阶段 10：小米 13 应用内性能 Profile

日期：2026-11-14。项目：`E:\work\ai\aaa-ncnn-vosr`。本阶段添加性能插桩，采集真实应用 UID 下的 NPU、GPU 和图像处理流水线；没有截屏或访问屏幕像素。

主结果来自手机解锁、VOSR 处于前台的采样：NPU 常驻单块推理中位数 **1.588 秒**，整次复原中位数 **2.231 秒**。GPU 兼容路径常驻单块推理均值 **38.550 秒**，整次复原约 **39.354 秒**。本轮没有修改模型精度、原图 patch 策略或模型权重。

测试固定使用自然尺寸 128×128 的 PNG，4× 输出为 512×512，恰好一个输出 patch，seed=42。输入文件为 `tests/fixtures/mobile/profile-128.png`，没有通过缩小其它测试图制造单块样例。其它尺寸依旧按 512 输出块、448 步长和 64 重叠执行。本报告的单块数字不能直接视为任意整图耗时。

设备为小米 13 / SM8550 / Adreno 740 / Android 13，应用包名 `ai.vosr.vosr_restore`，UID 10233。模型为 VOSR 0.5B。NPU 使用 QNN HTP v73，FP16 计算、FP32 输入输出；GPU 使用 ncnn 20260526 兼容路径，FP16 权重文件、FP32 计算，关闭 Winograd、subgroup 和 cooperative matrix。CPU 回退的运算没有被称为 GPU 内核执行。

| 前台采样 | 次数 | 单块原生推理 | 整次复原 | 模型校验 / 会话初始化 |
| --- | ---: | ---: | ---: | ---: |
| NPU 常驻，详细 QNN 采集关闭 | 5 | 中位 1.588 秒；范围 1.577–1.591 秒 | 中位 2.231 秒；范围 2.227–2.242 秒 | 两项均 0 |
| NPU 从 GPU 切回后首次建立会话 | 1 | 1.585 秒 | 6.615 秒 | 2.097 / 1.813 秒 |
| GPU 常驻，启用 CPU Reduction 计时 | 2 | 平均 38.550 秒；范围 38.481–38.620 秒 | 平均 39.354 秒 | 两项均 0 |
| GPU 切换后首次建立会话，详细计时关闭 | 1 | 38.741 秒 | 65.137 秒 | 2.021 / 22.755 秒 |

这些“首次”指需要重新建立该后端会话；没有重启手机、清空文件缓存或固定硬件频率。整次耗时包含校验、旧后端释放、新会话准备和处理，不包含用户选图和之后保存到相册。样本数较小，不报告 p95/p99 或稳定温度下的性能结论。9 个前台样本完成时的电源状态均为 `Awake`，应用 UID 均为 `2 (TOP)`；这是采样边界检查，不是全程屏幕录像。

NPU 的 5 次前台常驻采样中，各网络阶段的平均同步调用耗时如下。各阶段总和与完整原生调用的差值还包含输入输出拷贝、性能投票和调用封装。

| 网络阶段 | NPU 平均 | GPU 前台两次平均 |
| --- | ---: | ---: |
| VAE encoder | 212.6 ms | 6,584.7 ms |
| DINO | 87.7 ms | 3,106.1 ms |
| DiT prepare | 7.2 ms | 189.3 ms |
| DiT blocks 00–03 | 150.3 ms | 3,298.6 ms |
| DiT blocks 04–07 | 149.2 ms | 3,289.8 ms |
| DiT blocks 08–11 | 148.4 ms | 3,302.9 ms |
| DiT blocks 12–15 | 149.3 ms | 3,247.3 ms |
| DiT blocks 16–19 | 150.2 ms | 3,252.3 ms |
| DiT blocks 20–23 | 150.7 ms | 3,317.5 ms |
| DiT blocks 24–27 | 150.6 ms | 3,250.1 ms |
| DiT finish | 5.2 ms | 8.2 ms |
| VAE decoder | 216.9 ms | 5,689.5 ms |

NPU 的 28 层 DiT 合计约 **1.049 秒**，占完整原生调用约 66%；VAE 编码、解码合计约 **0.430 秒**。这些数值包含各自 QNN 同步调用的主机开销，不能全部称为纯硬件时间。

| NPU 前台流水线部分 | 5 次平均 |
| --- | ---: |
| 区域解码和双三次缩放 | 228.6 ms |
| RGB 归一化 | 2.7 ms |
| DINO 图像缩放和归一化 | 308.2 ms |
| 噪声生成 | 1.0 ms |
| JNI 调用整体 | 1,589.1 ms |
| patch 行缓存写入 | 10.6 ms |
| 拼接及 PNG 行输出 | 85.4 ms |
| 其中 PNG 编码及写入 | 37.4 ms，已包含在上一行 |

两次图像缩放合计约 **536.8 ms**，是原生推理以外的主要成本。NPU 原生输入拷贝均值约 0.9 ms、输出拷贝约 3.9 ms，当前不是主要瓶颈。不能把嵌套的 `pngEncodeWrite` 和 `stitchAndPng` 重复相加；脚本会导出未归属余量，避免强行让所有计时项完全相等。

最早的采集期间手机进入 `Dozing / TOP_SLEEPING`：NPU 原生推理仍为 1.56–1.63 秒，但区域解码/缩放增至约 0.84 秒，DINO 预处理增至约 1.30 秒，整次处理达到约 4.0 秒。GPU 的两次休眠详细样本为 59.77–60.20 秒，和前台组分开保存。电源状态、采集模式及温度都可能影响结果；本轮没有通过控制实验分离各因素，不能把全部差值归因于休眠或插桩。

GPU 的 85 个强制 CPU Reduction 在前台两次均调用 85 次，其函数执行累计分别为 **61.23 ms、64.53 ms**。该计时仅包围 CPU `forward` 运算，包含其内部并行等待，但不包含前后张量上传、下载和 Vulkan 队列同步。不能将其当成 CPU 回退路径的全部成本。

Perfetto 显示，前台 GPU 常驻两次应用所有线程合计运行的 CPU 时间分别为 **28.231、28.203 秒**；NPU 对应的应用 CPU 时间约 **28–36 ms**。多线程 CPU 时间是各线程实际占用 CPU 的时间之和，可以超过墙钟时间；GPU 主调用线程的睡眠也可能包含等待 CPU 工作线程，不能据此换算 GPU 利用率。

据此进一步包围 ncnn 中所有 `support_vulkan=false` 的层，确认整条 GPU 路径实际上执行了 **134 次 CPU MatMul 和 85 次 CPU Reduction**。另一次休眠诊断组中，常驻完整推理为 42.575 秒，其中 MatMul 的 CPU 函数墙钟时间合计 **6.478 秒**，Reduction 为 **77.25 ms**；子图输出的有限数值扫描合计 **28.44 ms**。134 次 MatMul 分布于 VAE、DINO 和 DiT，涉及注意力矩阵乘法；不能把这条路径称为全图纯 GPU。

该组关闭详细采集、复用同一引擎的控制样本为 43.644 秒，像素输出与详细采集样本一致。诊断组为 `Dozing`，不得将 6.478 秒直接放入前台 38.550 秒的时间账本；采集期间没有控制频率与温度，控制样本也不能给出精确的插桩开销百分比。所有 CPU 层函数计时均不包含进入函数前后的 Vulkan 传输、排队和同步。完整分层结果为 `cpu-layers.csv`。

同一诊断样本的 trace 中，应用各线程合计 CPU 时间为 40.873 秒，134 个 MatMul 区间内的应用 CPU 时间合计 25.380 秒。MatMul 内部使用多线程，因此该值高于其 6.478 秒函数墙钟时间。剩余 CPU 时间和 GPU 等待不能全部归给 MatMul；本轮没有用 GPU 硬件时间戳或驱动调用栈进一步分离传输、提交、驱动 CPU 工作和 GPU 内核耗时。

详细 QNN 硬件事件与延迟基准分别采集。一次对全部 12 个子图开启详细采集时，后续 4 个子图返回 `QNN_GRAPH_ERROR_SET_PROFILE (6006)`。现改为每个 context 使用独立 profile handle，并在该错误时记录失败后以普通模式重试，保留已采事件和正常图像输出。再单独从第 7 号子图开始采集，补齐 `dit_blocks_16/20/24` 和 decoder 的硬件事件。12 个阶段的成功事件来自两次明确标记的运行，**没有将这两次运行拼成一次端到端延迟**。

硬件 `Accelerator (execute)` 事件（类型 3004，单位微秒）中，encoder 为 209.875 ms、DINO 为 83.341 ms、四层 DiT 组约 144.3–146.8 ms、decoder 为 215.444 ms。原始 RPC、包含等待/排除等待、节点周期事件都保存在 CSV。嵌套事件不直接求和；cycles 不使用假定频率换成毫秒。QNN 优化后节点的名字也不能直接当作未融合 ONNX 单算子的边界。

本阶段代码增加 Android Trace 标记、Kotlin 流水线分段计时、JNI 拷贝/初始化计时、QNN 事件导出、ncnn CPU 层计时和自动采集/分析脚本。详细 profile 仅由 debug 测试入口显式打开；正常界面不启用详细 QNN 或逐层事件。原有模型和数值处理保持一致。

下一步优化的优先级有了依据：NPU 应先检查 CPU 双三次缩放及 DINO 预处理约 0.54 秒的成本，然后评估占原生调用约三分之二的 DiT；当前只有数毫秒的 JNI 拷贝和约 37 ms 的 PNG 编码不宜优先投入。GPU 应先解决 134 个 MatMul 的 CPU 回退及其跨设备传输、同步，再评估 FP16 算术等策略。更换运行库、算子或精度都需要重新做数值校验；本阶段只定位，没有实施这些运算优化。

所有 NPU 样本与插桩前 APK 的固定输入输出逐像素一致；GPU 对该 NPU 基准为 **54.4177 dB**，这是实现一致性检查，不是超分图像质量评分。前台 9 个完整 patch 和 108 个网络阶段均在 trace 中找到。Perfetto 仍记录了 150 个 systrace 解析警告，因此不声称系统 trace 完全无损；应用自己的阶段数量与 JSON 已独立校验。早期 trace 在 GPU 中途结束，只用于完整区间分析，未将未结束的 slice 当作耗时。

本地证据目录为 `artifacts/profiling/stage10/`：

仓库中另保留一份 [JSON 测量摘要](benchmarks/xiaomi13-2026-11-14.json)，包含前台逐次耗时、12 阶段数据及 CPU 回退诊断。下面的完整 trace、图像和日志仅保存在本地。

- `samples.json`、各样例 `result.json` / `timing.json` / `output.png`：完整原始结果及数值对照。
- `application-summary.csv`、`stage-wall.csv`、`cpu-reductions.csv`：流水线、12 阶段和归约算子耗时。
- `cpu-layers.csv`、`gpu-all-cpu.perfetto-trace`：全部 CPU 回退层和关闭详细采集的控制样本。
- `qnn-graphs.csv`、`qnn-events.csv`：硬件事件和失败/重试记录。
- `foreground.perfetto-trace`、`foreground-slices.csv`、`foreground-thread-states.csv`：前台 CPU 调度和应用阶段。
- `*-thermal.txt`、`*-power.txt`、`*-uid-state.txt`：测量条件快照。
- `verification.json`：输出、前台状态、会话复用、事件完整性以及 trace 文件校验和。

Perfetto Python 包与 protobuf 安装在独立的 `E:\work\ai\tools\profiling-env`，没有修改模型转换环境。Trace Processor 为官方 v59.1，路径 `E:\work\ai\tools\perfetto\trace_processor_shell.exe`，SHA-256 为 `3acc5aca1c764541033a82ba3bb0a47524b5c3c0aa194eea8d9a7b0cdb346bdc`，已对照发行清单。分析使用本地 SQL CLI；参考 [Perfetto 官方分析文档](https://perfetto.dev/docs/analysis/trace-processor-python)。

最终保留 25 个成功样本（另有插桩前基准及失败尝试），输出、前台状态、CPU 回退次数、详细采集开关一致性均由 `scripts/verify-mobile-profile.py` 校验通过。Flutter 分析无问题，最终 JNI / Kotlin / APK 构建成功，APK 中 6 个原生库与构建输入逐字节匹配，两个模型 ZIP 的 SHA-256 未变。APK 校验和已更新到 `deliverables/SHA256.json`；手机已安装最终版本。阶段 9 的数值和时延属于此前验收记录，新的原生插桩验证记录在本阶段。

复现应先保留插桩前 APK 并以 `--baseline-only` 保存基准，再安装 profile 版本；已有数据不要直接覆盖。所有工具、缓存和证据均在 `E:\work\ai`。

```powershell
. .\scripts\env.ps1
.\.venv\Scripts\python.exe scripts\profile-mobile.py --foreground
.\.venv\Scripts\python.exe scripts\profile-mobile.py --detailed-npu --profile-start-stage 0
.\.venv\Scripts\python.exe scripts\profile-mobile.py --detailed-npu --profile-start-stage 7
.\.venv\Scripts\python.exe scripts\profile-mobile.py --all-cpu-gpu
.\.venv\Scripts\python.exe scripts\analyze-mobile-profile.py
.\.venv\Scripts\python.exe scripts\verify-mobile-profile.py
.\.venv\Scripts\python.exe scripts\package-mobile.py --apk-only
```
