# ReOldify

**A community-driven continuation of [DeOldify](https://github.com/jantic/DeOldify)** — Deep Learning based colorization and restoration of old images and video.

> DeOldify was created by [Jason Antic](https://github.com/jantic) in 2018 and archived in October 2024 after 6 incredible years. ReOldify picks up where it left off, with modernized dependencies, a proper CLI, and continued development.

![Migrant Mother](https://i.imgur.com/Bt0vnke.jpg)
*"Migrant Mother" by Dorothea Lange (1936) — colorized by DeOldify/ReOldify*

---

## What's New in ReOldify

- **Modernized stack**: Python 3.10+, PyTorch 2.x, updated Pillow & OpenCV
- **CLI tool**: `reoldify colorize photo.jpg` — no Jupyter notebook needed
- **Fixed dependencies**: Resolved ffmpeg/ffmpeg-python conflicts
- **Modern packaging**: `pyproject.toml` with proper dependency management
- **Cross-platform**: Windows, Linux, macOS support

## Quick Start

### Installation

```bash
pip install git+https://github.com/pattex67/ReOldify.git
```

Or clone and install locally:

```bash
git clone https://github.com/pattex67/ReOldify.git
cd ReOldify
pip install -e .
```

### CLI Usage

```bash
# Colorize an image (artistic model - most colorful)
reoldify colorize photo.jpg

# Use the stable model (better for portraits & landscapes)
reoldify colorize photo.jpg --model stable

# Colorize a video
reoldify colorize video.mp4

# Adjust render quality (higher = better but slower, default: 35)
reoldify colorize photo.jpg --render-factor 45

# Specify output path
reoldify colorize photo.jpg -o colorized_photo.jpg
```

### Python API

```python
from deoldify.visualize import get_image_colorizer

# Artistic model (most vibrant colors)
colorizer = get_image_colorizer(artistic=True)
result_path = colorizer.plot_transformed_image(
    path="old_photo.jpg",
    render_factor=35,
    watermarked=False
)

# Stable model (best for portraits)
colorizer = get_image_colorizer(artistic=False)
result_path = colorizer.plot_transformed_image(path="portrait.jpg")
```

### Jupyter Notebooks

The original notebooks are still available for interactive use:

- `ImageColorizerColab.ipynb` — Image colorization (artistic)
- `ImageColorizerColabStable.ipynb` — Image colorization (stable)
- `VideoColorizerColab.ipynb` — Video colorization

## Pretrained Weights

Download the weights and place them in the `models/` folder:

### Generator Weights (required for inference)

| Model | Use Case | Download |
|-------|----------|----------|
| **Artistic** | Most colorful, best for general images | [Download](https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth) |
| **Stable** | Best for portraits & landscapes, fewer artifacts | [Download](https://www.dropbox.com/s/axsd2g85uyixaho/ColorizeStable_gen.pth?dl=0) |
| **Video** | Optimized for smooth, flicker-free video | [Download](https://data.deepai.org/deoldify/ColorizeVideo_gen.pth) |

## Three Models Explained

- **Artistic** — Highest quality, most vibrant colors. Uses ResNet34 backbone with deep U-Net decoder. Best for general-purpose colorization but may need render_factor tuning.

- **Stable** — Best for portraits and nature. Uses ResNet101 backbone with wide U-Net decoder. Fewer artifacts and "zombie" effects, slightly less colorful.

- **Video** — Optimized for temporal consistency and flicker-free video output. Same architecture as stable but trained for smoothness.

## Technical Background

ReOldify uses the **NoGAN** training technique (developed by Jason Antic), which combines the benefits of GAN training with minimal direct GAN training time. The approach:

1. Pretrain the generator with perceptual loss (VGG16-based)
2. Train the critic on generated vs. real images
3. Brief GAN training (1-3% of ImageNet data) to close the realism gap

This produces colorful, artifact-free results — especially important for video consistency.

## Hardware Requirements

- **Inference**: GPU with 4GB+ VRAM recommended (CPU works but slower)
- **Training**: GPU with 11GB+ VRAM (e.g., RTX 3080 or better)

## Original Project

ReOldify is a fork of [jantic/DeOldify](https://github.com/jantic/DeOldify), originally created by Jason Antic. All original code is under the MIT license. We are grateful for his pioneering work in image colorization.

Related projects from the DeOldify ecosystem:
- [sd-webui-deoldify](https://github.com/SpenserCai/sd-webui-deoldify) — Stable Diffusion Web UI plugin
- [DeOldify.NET](https://github.com/ColorfulSoft/DeOldify.NET) — Windows GUI (no GPU required)
- [DeOldify-on-Browser](https://github.com/akbartus/DeOldify-on-Browser) — ONNX-based browser implementation

## Contributing

Contributions are welcome! Areas where help is especially appreciated:

- Integrating modern colorization techniques
- ONNX/TensorRT export for fast inference
- Improving color accuracy
- Adding batch processing support
- Documentation and examples

## License

MIT License — see [LICENSE](LICENSE) for details.

All pretrained weights are released under the MIT license by the original DeOldify authors.
