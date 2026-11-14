# 第三方组件说明

此项目将以下组件用于本地研究和 Android 部署。保留上游版权、许可和模型使用条款。

| 组件 | 官方来源与条款入口 |
| --- | --- |
| VOSR 源码、单步 checkpoint、LightDecoder | https://github.com/cswry/VOSR ，Apache-2.0；原始 LICENSE 在 third_party/VOSR-main/LICENSE |
| Stable Diffusion 2.1 VAE | https://huggingface.co/stabilityai/stable-diffusion-2-1-base ，上游 OpenRAIL++ 条款；VOSR 模型卡保存于 evidence/MODEL_CARD.md |
| DINOv2 | https://github.com/facebookresearch/dinov2 ，下载源码内保留 LICENSE 和各文件头 |
| 上游继承的 DiT / EVA / RMSNorm 等代码 | 保留 third_party/VOSR-main/models 中的原始版权与许可文件头；以对应上游条款为准 |
| Qualcomm QAIRT / QNN | 官方 SDK 2.44.0.260225；完整 SDK 许可文件在 tools/qairt/2.44.0.260225/LICENSE.pdf（相对 E:/work/ai） |
| Flutter / Dart | https://github.com/flutter/flutter ，保留安装 SDK 内许可；应用可使用 Flutter 的许可查看功能 |
| AndroidX Core / ExifInterface | https://developer.android.com/jetpack/androidx ，Apache-2.0；用于相机文件授权和 EXIF 解析 |
| PyTorch、diffusers、ONNX、ONNX Runtime 等 | 独立环境保留各发行包许可；精确版本在 evidence/requirements-*.lock |
| ncnn / pnnx / glslang | https://github.com/Tencent/ncnn ，BSD-3-Clause；移动端 ncnn 固定 20260526 / e54f7b1，pnnx 固定 20261020，依赖保留各自许可 |

本文件是组件索引，不取代各组件的原始许可。模型与运行库的重新分发须保留并遵守适用的上游条款。
