# Derived from DDColor (Apache-2.0) — https://github.com/piddnad/DDColor
"""DDColor inference pipeline — loads model from HuggingFace and colorizes images."""
import logging
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

logger = logging.getLogger(__name__)

# HuggingFace model repo IDs
MODEL_REPOS = {
    "ddcolor": "piddnad/ddcolor_artistic",
    "ddcolor-tiny": "piddnad/ddcolor_paper_tiny",
}

# Model configs
MODEL_CONFIGS = {
    "ddcolor": dict(model_size="large", input_size=512, num_queries=100, num_scales=3, dec_layers=9),
    "ddcolor-tiny": dict(model_size="tiny", input_size=512, num_queries=100, num_scales=3, dec_layers=9),
}


def _download_weights(model_name: str) -> str:
    """Download DDColor weights from HuggingFace Hub. Returns local path."""
    from huggingface_hub import hf_hub_download

    repo_id = MODEL_REPOS[model_name]
    # The HF repos store weights as pytorch_model.pt or similar
    # Try common filenames
    for filename in ["pytorch_model.pt", "pytorch_model.pth", "model.pth", "model.safetensors"]:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename)
            logger.info(f"Downloaded {model_name} weights from {repo_id}/{filename}")
            return path
        except Exception:
            continue

    # If none of the common names work, try listing files
    from huggingface_hub import list_repo_files
    files = list_repo_files(repo_id)
    for f in files:
        if f.endswith(('.pt', '.pth', '.bin')):
            path = hf_hub_download(repo_id=repo_id, filename=f)
            logger.info(f"Downloaded {model_name} weights from {repo_id}/{f}")
            return path

    raise RuntimeError(f"Could not find model weights in {repo_id}. Files: {files}")


def build_model(model_name: str = "ddcolor", device=None):
    """Build a DDColor model and load pretrained weights."""
    from .arch import DDColor

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = MODEL_CONFIGS[model_name]
    encoder_name = "convnext-t" if config["model_size"] == "tiny" else "convnext-l"
    input_size = config["input_size"]

    model = DDColor(
        encoder_name=encoder_name,
        decoder_name="MultiScaleColorDecoder",
        input_size=[input_size, input_size],
        num_output_channels=2,
        last_norm="Spectral",
        do_normalize=False,
        num_queries=config["num_queries"],
        num_scales=config["num_scales"],
        dec_layers=config["dec_layers"],
    )

    weights_path = _download_weights(model_name)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "params" in ckpt:
        state_dict = ckpt["params"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    logger.info(f"DDColor model '{model_name}' loaded on {device}")
    return model


class ColorizationPipeline:
    """DDColor colorization pipeline.

    Input: BGR uint8 (OpenCV format) or RGB PIL Image
    Output: BGR uint8 (OpenCV format)
    """

    def __init__(self, model, *, input_size: int = 512, device=None):
        self.input_size = int(input_size)
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()

    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        """Colorize a BGR uint8 image. Returns BGR uint8."""
        ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with ctx():
            if img_bgr is None:
                raise ValueError("Input image is None")

            height, width = img_bgr.shape[:2]

            # Original luminance at full resolution
            img = (img_bgr / 255.0).astype(np.float32)
            orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]

            # Resize -> grayscale LAB -> back to RGB for model input
            img_resized = cv2.resize(img, (self.input_size, self.input_size))
            img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
            img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
            img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

            tensor_gray_rgb = (
                torch.from_numpy(img_gray_rgb.transpose((2, 0, 1)))
                .float().unsqueeze(0).to(self.device)
            )

            # Model predicts AB channels
            output_ab = self.model(tensor_gray_rgb).cpu()

            # Resize AB to original size, combine with original L
            output_ab_resized = (
                F.interpolate(output_ab, size=(height, width))[0]
                .float().numpy().transpose(1, 2, 0)
            )
            output_lab = np.concatenate((orig_l, output_ab_resized), axis=-1)
            output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)

            return (output_bgr * 255.0).round().astype(np.uint8)
