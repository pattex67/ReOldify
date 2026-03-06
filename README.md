# ReOldify

**A community-driven continuation of [DeOldify](https://github.com/jantic/DeOldify)** — Deep Learning based colorization and restoration of old images and video.

> DeOldify was created by [Jason Antic](https://github.com/jantic) in 2018 and archived in October 2024. ReOldify picks up where it left off, with a modernized stack, a new state-of-the-art colorization engine (DDColor), and continued development.

---

## What's New in ReOldify

- **DDColor engine** (ICCV 2023): State-of-the-art colorization with significantly richer, more accurate colors
- **Modernized stack**: Python 3.10+, PyTorch 2.x, updated dependencies
- **CLI tool**: `reoldify colorize photo.jpg --model ddcolor --cpu`
- **REST API**: FastAPI server to colorize via HTTP
- **Docker**: One-command deployment with Docker Compose
- **CI/CD**: GitHub Actions for linting and testing
- **Modern packaging**: `pyproject.toml` with proper dependency management

## Quick Start

### Installation

```bash
git clone https://github.com/pattex67/ReOldify.git
cd ReOldify
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

### CLI Usage

```bash
# Colorize with DDColor (best quality, weights auto-downloaded)
reoldify colorize photo.jpg --model ddcolor --cpu

# DDColor tiny (faster, lighter)
reoldify colorize photo.jpg --model ddcolor-tiny --cpu

# Legacy DeOldify models
reoldify colorize photo.jpg --model artistic --cpu
reoldify colorize photo.jpg --model stable --cpu

# Colorize a video
reoldify colorize video.mp4 --cpu

# Adjust render quality (higher = better but slower, default: 35)
reoldify colorize photo.jpg --model ddcolor --render-factor 45

# Specify output path
reoldify colorize photo.jpg -o colorized_photo.jpg --model ddcolor --cpu
```

### Python API

```python
from deoldify import device as device_settings
from deoldify.device_id import DeviceId
device_settings.set(DeviceId.CPU)

# DDColor (recommended)
from deoldify.visualize import get_ddcolor_image_colorizer
colorizer = get_ddcolor_image_colorizer(model_name="ddcolor")
result = colorizer.get_transformed_image("old_photo.jpg", render_factor=35, watermarked=False)
result.save("colorized.jpg")

# Legacy DeOldify
from deoldify.visualize import get_image_colorizer
colorizer = get_image_colorizer(artistic=True)
result = colorizer.get_transformed_image("old_photo.jpg", render_factor=35, watermarked=False)
result.save("colorized.jpg")
```

### REST API

```bash
pip install fastapi uvicorn python-multipart
python -m deoldify.api
```

Then colorize images via HTTP:

```bash
curl -X POST http://localhost:8000/colorize \
  -F "file=@old_photo.jpg" \
  -F "model=ddcolor" \
  -o colorized.png
```

### Docker

```bash
docker compose up --build
```

## Available Models

| Model | Engine | CLI flag | Size | Best for |
|-------|--------|----------|------|----------|
| **DDColor** | DDColor (2023) | `--model ddcolor` | 912 MB | Best quality, general use |
| **DDColor Tiny** | DDColor (2023) | `--model ddcolor-tiny` | 220 MB | Fast / CPU inference |
| **Artistic** | DeOldify (2019) | `--model artistic` | 243 MB | Vibrant colors (legacy) |
| **Stable** | DeOldify (2019) | `--model stable` | 243 MB | Portraits (legacy) |
| **Video** | DeOldify (2019) | `--model video` | 243 MB | Video colorization |

**DDColor** weights are auto-downloaded from HuggingFace on first use.

**DeOldify** weights must be downloaded manually to the `models/` folder:

| Model | Download |
|-------|----------|
| Artistic | [ColorizeArtistic_gen.pth](https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth) |
| Stable | [ColorizeStable_gen.pth](https://www.dropbox.com/s/axsd2g85uyixaho/ColorizeStable_gen.pth?dl=0) |
| Video | [ColorizeVideo_gen.pth](https://data.deepai.org/deoldify/ColorizeVideo_gen.pth) |

## Technical Background

### DDColor (ICCV 2023)
DDColor uses a dual-decoder architecture: a pixel decoder restores spatial resolution while a query-based color decoder refines learnable color tokens via cross-attention. This produces richer, more accurate colors with fewer artifacts than previous approaches. Uses ConvNeXt as backbone encoder.

### DeOldify (2019)
DeOldify uses the **NoGAN** training technique, combining GAN training benefits with minimal direct GAN training time. Uses a U-Net architecture with ResNet34 (artistic) or ResNet101 (stable) backbones.

## Hardware Requirements

- **Inference**: GPU recommended for speed. CPU works fine (especially with `ddcolor-tiny`)
- **Training**: GPU with 11GB+ VRAM

## Credits

- [DeOldify](https://github.com/jantic/DeOldify) by Jason Antic (MIT License)
- [DDColor](https://github.com/piddnad/DDColor) by Xiaoyang Kang et al. (Apache-2.0 License)

## Contributing

Contributions are welcome! Areas where help is appreciated:

- Improving color accuracy
- ONNX/TensorRT export for fast inference
- Adding batch processing support
- Video support with DDColor

## License

MIT License — see [LICENSE](LICENSE) for details.

DDColor architecture code is derived from the [DDColor project](https://github.com/piddnad/DDColor) under the Apache-2.0 license.
