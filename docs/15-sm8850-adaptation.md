# 阶段 15：SM8850 / Adreno 840 适配

日期：2026-11-15。

设备实读为 OPPO PMA110，SoC 为 SM8850，Android 17 / API 37，4 KiB 页面。GPU 为 Adreno 840，`maxStorageBufferRange=2147483647`，可容纳 1024 VAE 的 512 MiB 特征图。当前没有根据型号推断零售名称。

832 工作尺寸、512 输入块、96 输入重叠、416 输入步长及固定 2× 不变。832×624 为四块，每块输出 1024×1024，最终仍融合回原图尺寸。

## NPU

QAIRT 2.44 的 `QnnTypes.h` 定义 SM8850 的 SoC ID 为 87，SDK 设备表对应 HTP v81。新模型复用已通过数值检查的 FP16 DLC，重新编译为 v81 context，VTCM 使用目标最大值。旧 v73 context 不用于新芯片。

v81 使用 extended uDMA 的 far mapping，并共享 spill/fill 工作区。全部 12 个 context 常驻的尝试在 `dit_blocks_12` 加载处返回 1007，未通过当前固件的应用内测试；最终仍常驻五个 context，七组 DiT 顺序加载、释放，最多同时保留六个。SM8550 沿用原策略。两类模型保存在不同目录，导入时同时校验芯片、后端、尺寸与文件哈希。

```powershell
. .\scripts\qnn-env.ps1
E:\work\ai\vosr-mobile\.venv-qnn\Scripts\python.exe conversion\build_qnn.py --onnx artifacts\onnx-1024 --native-vae-onnx artifacts\onnx-qnn-1024 --dlc-source artifacts\qnn-1024 --output artifacts\qnn-sm8850-1024 --soc-model 87 --dsp-arch v81 --vtcm-mb 0 --extended-udma --workers 2
```

v81 的 Stub、Skel 和 DSP 依赖库从同版本官方 SDK 补齐，与原 v73 运行库一并打入 APK。若使用的 SDK 是此前的精简提取版，可按固定路径补齐：

```powershell
. .\scripts\env.ps1
.\.venv\Scripts\python.exe scripts\remote_zip.py https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.44.0.260225/v2.44.0.260225.zip --listing evidence\qairt-v81-list.json --cache-dir ..\cache\qairt-ranges --extract ..\tools --patterns conversion\qairt-runtime-v81-patterns.json
```

工具只请求需要的 ZIP 字节块，检查 HTTP Range、长度、ZIP CRC 和提取路径。已安装运行库的大小、SHA-256 另存于 `conversion/qairt-runtime-v81.json`。SDK 仍需遵守原许可。

## GPU

1024 VAE 是否回退 CPU 改为读取实际 Vulkan 缓冲区上限。PMA110 上 VAE 已使用 GPU，继续关闭 Winograd 和大尺寸 VAE 的 SGEMM，避免大临时缓冲；MatMul、Reduction 等 CPU 算子回退保留。网络仍按子图加载和释放，同时只保留一个 Net。小米 13 的 128 MiB 上限仍选择 CPU VAE。

权重没有改变，新 GPU 包单独记录 Adreno 840 的验证结果与配置。界面从设备读取后端说明，不再固定显示骁龙 8 Gen 2。

## 验证范围

本阶段使用公开参考张量和合成 EXIF 测试图，不截屏、不读取私人照片。数值基准为同一输入的 FP32 ONNX 输出。

| 检查 | 结果 |
| --- | --- |
| SDK 的 12 个 v81 子图顺序执行 | PSNR 60.16 dB，各子图 NRMSE <0.05 |
| APK 内完整 NPU JNI | PSNR 60.16 dB，FP16 计算、FP32 输入输出 |
| APK 内完整 GPU JNI | PSNR 82.00 dB，168.928 秒；FP16 权重、FP32 计算，VAE 使用 Vulkan |
| NPU 四块整图 | 4000×3000 → 832×624 → 4 块 → 4000×3000，保存到 MediaStore 并确认尺寸 |
| 原生图像与配置回归 | 4 块几何、8 种 EXIF 方向、严格变焦门槛、金字塔融合和双芯片模型匹配均通过 |
| Flutter | 静态分析通过，4 项测试通过 |
| 交付包 | 两个新模型 ZIP 的文件哈希、APK 内 10 个项目原生库与本地构建逐一匹配 |

NPU 整图功能验收耗时 **43.823 秒**，包含模型校验、初始化、预处理、推理、拼接和融合。四块原生调用分别为 **7272.72、6903.56、7114.39、6970.53 ms**，合计 28.261 秒；原尺寸融合为 4.226 秒。末块图执行合计 4.167 秒，context 加载 2.771 秒、释放 27 ms。原生调用包含动态加载，不能当作纯图执行时间。

这些是功能验收的单次样本，没有固定温度和频率。另一次独立 NPU JNI 记录耗时 69.056 秒，其中 context 加载占 64.401 秒，存在明显加载波动；不把这一轮数据当成稳定性能基准。四块整图结束时电源状态为 Awake。完整计时见 [JSON 记录](benchmarks/sm8850-2026-11-15.json)。

GPU 验证运行了一次完整的 12 子图 JNI，没有运行四块整图。公共的分块、融合与保存流程由上述 NPU 整图验收覆盖。新手机系统相机的实拍 EXIF 尚未验证；严格门槛的合成样例通过，缺少变焦字段仍拒绝处理。

复现本轮验收可运行：

```powershell
. .\scripts\env.ps1
.\.venv\Scripts\python.exe scripts\test-photo-backends.py --serial <设备序列号> --backends npu --deploy
.\.venv\Scripts\python.exe scripts\test-photo-backends.py --serial <设备序列号> --backends gpu --deploy --raw-only
.\.venv\Scripts\python.exe scripts\package-mobile.py --soc SM8850
```

脚本按设备 SoC 选择模型和证据目录。设备需解锁并保持应用前台，日志与输出保留在 `artifacts/photo-tests-sm8850-1024/`。SDK 数值验证使用 `validate_device.py` 的 `--dsp-arch v81 --target-soc SM8850 --soc-model-id 87 --extended-udma`，设备测试目录通过 `--remote` 单独指定；目录内需准备同版本 qnn-net-run 和运行库。

## 安装与交付

APK 已安装到 PMA110，NPU 和 GPU 模型已复制并校验。NPU 使用 `models-sm8850-1024`，GPU 使用 `models-gpu-1024`。安装脚本读取 SoC，自动选择本机模型目录，拒绝复制不匹配的 NPU context。

新增本地交付文件：

- `vosr-0.5b-sm8850-htp-v81-1024-models.zip`
- `vosr-0.5b-sm8850-ncnn-vulkan-1024-models.zip`

同一个 APK 包含 v73、v81 运行库，原有 512 输出规格及小米 13 的 1024 输出模型继续保留。文件校验和见 [SHA256.json](../deliverables/SHA256.json)；大模型、APK 和厂商库不进入 Git。
