# ncnn-VOSR

[VOSR](https://github.com/cswry/VOSR) 单步超分的 ncnn 部署。包含原有的桌面 1.4B 实现，以及支持 QNN NPU / ncnn Vulkan 的 Flutter Android 应用。

| 实现 | 模型 | 后端 | 文档 |
| --- | --- | --- | --- |
| Android | VOSR 0.5B | Qualcomm QNN HTP、ncnn Vulkan | [安装、构建和转换](docs/android.md) |
| 桌面 Python / C++ | VOSR2 1.4B | ncnn CPU、Vulkan | [桌面用法](docs/desktop.md) |

两套模型的 VAE、DINO 和张量形状不同，模型文件不能混用。

## Android

应用支持选图、2× / 4× 复原、原图对比和保存到相册。输入按原始分辨率读取，处理 EXIF 方向后切成重叠 patch；输出块为 512×512，步长 448，重叠区域做高斯融合，逐行写 PNG。预览缩略图不参与推理。

另有[系统相机拍照复原](docs/12-camera-restoration.md)：严格要求 EXIF 数字变焦 ≥10×；拍照原图等比例缩放到最长边 832，固定 2×、重叠 96 处理，再缩回拍照原尺寸，通过高斯金字塔与原始 LR 融合。

NPU 和 GPU 各自常驻 12 个子图，同一后端连续处理图片时复用会话；切换后端时释放旧会话。

- **NPU**：QNN HTP v73，FP16 计算、FP32 输入输出。目前只验证并开放小米 13 / SM8550。
- **GPU**：ncnn Vulkan，FP16 权重存储、FP32 计算。小米 13 的兼容路径中，134 个 MatMul 和 85 个 Reduction 在 CPU 执行，速度明显慢于 NPU。其它手机尚未验证。

先安装 APK，再在应用里为所选后端导入对应模型 ZIP，选择照片后点击“开始复原”。当前 APK 和模型包保存在本地 `deliverables/`，没有上传到 Git；文件名及校验和见 [使用说明](deliverables/使用说明.md) 和 [SHA256.json](deliverables/SHA256.json)。源码构建见 [Android 文档](docs/android.md)。

## 小米 13 性能

128×128 输入，4× 输出，单个 512×512 patch；手机解锁且应用处于前台。

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
- [阶段文档索引](docs/STATUS.md)

`docs/history/` 是接入本仓库前的开发记录，其中的路径和性能数字保留了当时的状态。当前用法以 Android 文档为准。

项目许可见 [LICENSE](LICENSE)，模型及运行库的许可见 [第三方组件说明](THIRD_PARTY_NOTICES.md)。
