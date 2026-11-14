---
license: apache-2.0
pipeline_tag: image-to-image
tags:
  - image-super-resolution
  - image-restoration
---

# VOSR: A Vision-Only Generative Model for Image Super-Resolution

**A framework for native generative image restoration.**

Many recent generative super-resolution methods adapt pretrained text-to-image models. VOSR trains its generative backbone from scratch for super-resolution, using structural and visual semantic conditions from the low-resolution input. It requires no text prompts and reuses pretrained VAE and vision-encoder components.

Our goal is to build native generative models for image restoration that combine input fidelity, perceptual quality, and efficient inference. VOSR explores this direction through restoration-oriented guidance and both multi-step and one-step models.

## Available checkpoints

This repository provides:

- `VOSR2/`: VOSR 2.0, a 1.4B one-step model.
- `VOSR_0.5B_ms/`: 0.5B multi-step model.
- `VOSR_0.5B_os/`: 0.5B one-step model.
- `VOSR_1.4B_ms/`: 1.4B multi-step model.
- `VOSR_1.4B_os/`: 1.4B one-step model.

Supporting VAE, decoder, and visual-encoder files are also provided. See the license section for the terms that apply to each component.

`Qwen-Image-vae-2d/` is our 2D conversion of the released 3D causal VAE weights, with the encoder from [Wan2.1](https://github.com/Wan-Video/Wan2.1) and the image decoder fine-tuned by [Qwen-Image](https://arxiv.org/html/2508.02324v1#S2.SS3).

## Intended use and limitations

VOSR is intended for research and image-restoration applications, including single-image super-resolution.

Generative super-resolution may produce plausible details that are not fully supported by the input image. The output should not be treated as a faithful recovery of information absent from the input or as forensic evidence. Results may vary with the input image, degradation, checkpoint, and inference settings.

## Training and evaluation

The accompanying paper describes the training setup and evaluation protocols. It reports training on a filtered image collection of approximately 100 million images and using synthetic low-resolution/high-resolution pairs generated with Real-ESRGAN degradation. The training data is not included in this repository.

See the [paper](https://arxiv.org/abs/2604.03225) for evaluation results and further details.

## Inference

Use the inference code in the [VOSR project repository](https://github.com/cswry/VOSR). This repository uses VOSR-specific inference scripts; the generic `DiffusionPipeline.from_pretrained(...)` example shown by Hugging Face may not apply.

Clone the project and install its dependencies:

```bash
git clone https://github.com/cswry/VOSR.git
cd VOSR
pip install -r requirements.txt
```

Download the required checkpoint files from this Hugging Face repository into the project’s `preset/ckpts/` directory, preserving their folder names. For example, to run VOSR 2.0 on images in `preset/datasets/inp_data`:

```bash
python inference_vosr_onestep.py \
  -c preset/ckpts/VOSR2 \
  -i preset/datasets/inp_data \
  -o preset/results \
  -u 4
```

For inference with other checkpoints and options, see the [project README](https://github.com/cswry/VOSR).

## License

The Apache-2.0 license applies to VOSR project materials covered by the [project license](https://github.com/cswry/VOSR/blob/main/LICENSE), including project-authored checkpoints unless a release-specific notice says otherwise. Some bundled third-party components have their own terms; the Apache-2.0 license does not replace those terms.

| Files | Terms |
|---|---|
| VOSR checkpoints (`VOSR2/`, `VOSR_*/`) and project-authored decoder files | Apache-2.0 under the [project license](https://github.com/cswry/VOSR/blob/main/LICENSE), unless a release-specific notice says otherwise. |
| `Qwen-Image-vae-2d/` | Follow the applicable upstream terms of [Wan2.1](https://github.com/Wan-Video/Wan2.1) and [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image). |
| `stable-diffusion-2-1-base/` | Follow the [upstream Stable Diffusion 2.1 terms](https://huggingface.co/stabilityai/stable-diffusion-2-1-base), listed as OpenRAIL++. |
| `torch_cache/` | If this directory contains DINOv2 weights, follow the terms in the [DINOv2 repository](https://github.com/facebookresearch/dinov2). |

Please retain applicable upstream notices and comply with the terms for each third-party component.

## Citation

If you use VOSR in your research, please cite:

```bibtex
@inproceedings{wu2026vosr,
  title     = {VOSR: A Vision-Only Generative Model for Image Super-Resolution},
  author    = {Wu, Rongyuan and Sun, Lingchen and Zhang, Zhengqiang and Kong, Xiangtao and Zhao, Jixin and Wang, Shihao and Zhang, Lei},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```

## Links

- Code: [https://github.com/cswry/VOSR](https://github.com/cswry/VOSR)
- Paper:[https://arxiv.org/abs/2604.03225](https://arxiv.org/abs/2604.03225)
