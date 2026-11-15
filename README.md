# ncnn-VOSR

[VOSR](https://github.com/cswry/VOSR) 单步超分的 ncnn 部署。包含原有的桌面 1.4B 实现，以及支持 QNN NPU / ncnn Vulkan 的 Flutter Android 应用。

| 实现 | 模型 | 后端 | 文档 |
| --- | --- | --- | --- |
| Android | VOSR 0.5B | Qualcomm QNN HTP、ncnn Vulkan | [安装、构建和转换](docs/android.md) |
| 桌面 Python / C++ | VOSR2 1.4B | ncnn CPU、Vulkan | [桌面用法](docs/desktop.md) |

两套模型的 VAE、DINO 和张量形状不同，模型文件不能混用。

## Android

应用支持选图复原、原图对比和保存到相册。处理 EXIF 方向后，输入等比例调整到最长边 832，按 **512×512 输入块、96 输入重叠、416 输入步长**分块。固定 2× 推理，每块输出 1024×1024；832×624 实际运行 4 块。拼接后缩回原图尺寸，通过高斯金字塔与输入融合。

另有[系统相机拍照复原](docs/12-camera-restoration.md)：严格要求 EXIF 数字变焦 ≥10×，通过后使用同一处理流程。分块尺度及新模型说明见[阶段 14](docs/14-input-patch-512.md)。

移动端模型包含 12 个子图。NPU 常驻其中 5 个，DiT 按组加载、执行后释放，避免大尺寸模型耗尽 DSP 地址空间；GPU 每次加载一个子图，用完释放。同一后端可复用引擎及配置，切换后端会释放旧会话。

- **NPU**：QNN HTP，FP16 计算、FP32 输入输出。支持小米 13 / SM8550 / v73，以及 OPPO PMA110 / SM8850 / v81；模型包按芯片区分。
- **GPU**：ncnn Vulkan，FP16 权重存储、FP32 计算。按实际缓冲区上限选择 VAE 路径：小米 13 的 128 MiB 上限需要 CPU VAE，PMA110 的约 2 GiB 上限可使用 GPU VAE。部分 MatMul、Reduction 仍在 CPU 执行。

先安装 APK，再在应用里为所选后端导入对应模型 ZIP，选择照片后点击“开始复原”。当前 APK 和模型包保存在本地 `deliverables/`，没有上传到 Git；文件名及校验和见 [使用说明](deliverables/使用说明.md) 和 [SHA256.json](deliverables/SHA256.json)。源码构建见 [Android 文档](docs/android.md)。

PMA110 的 12MP 四块 NPU 功能验收约 43.8 秒，四块原生调用为 6.9–7.3 秒/块；GPU 单次原生调用约 169 秒。该轮没有控制温度和频率，记录及加载波动见 [SM8850 适配](docs/15-sm8850-adaptation.md)。

## 小米 13 性能

以下为阶段 10 的历史测量：128×128 输入，4× 输出，单个 512×512 输出块；手机解锁且应用处于前台。新 1024 输出规格须单独测量，见阶段 14。

| 后端 | 常驻推理 | 整次复原 | 样本 |
| --- | ---: | ---: | ---: |
| NPU | 1.588 秒 | 2.231 秒 | 5 次，中位数 |
| GPU 兼容路径 | 38.550 秒 | 39.354 秒 | 2 次，均值 |

推理时间包含编码、DINO、DiT、解码和原生输入输出拷贝；整次复原还包含图像预处理与拼接。上表不含首次模型加载，GPU 首次会话加载另约 23 秒。大图耗时随 patch 数增长，休眠和频率变化也会影响结果。

NPU 的主要计算成本在 DiT，约 1.05 秒；CPU 图像预处理另约 0.54 秒。GPU 的矩阵乘法回退和跨设备同步仍需优化。完整分段计时、测量条件和数值检查见 [性能报告](docs/10-performance-profile.md)。

## 开发记录

- [Android 架构与调试入口](app/README.md)
- [模型转换与仓库接入](docs/08-repository-integration.md)
- [双后端真机测试](docs/09-mobile-backend-acceptance.md)
- [性能 Profile](docs/10-performance-profile.md)
- [SM8850 / Adreno 840 适配](docs/15-sm8850-adaptation.md)
- [阶段文档索引](docs/STATUS.md)

`docs/history/` 是接入本仓库前的开发记录，其中的路径和性能数字保留了当时的状态。当前用法以 Android 文档为准。

项目许可见 [LICENSE](LICENSE)，模型及运行库的许可见 [第三方组件说明](THIRD_PARTY_NOTICES.md)。
