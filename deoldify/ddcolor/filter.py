"""DDColor filter adapter for ReOldify's IFilter interface."""
import cv2
import numpy as np
from PIL import Image as PilImage

from ..filters import IFilter
from .pipeline import ColorizationPipeline, build_model, MODEL_CONFIGS


class DDColorFilter(IFilter):
    """Wraps DDColor pipeline into ReOldify's IFilter interface."""

    def __init__(self, model_name: str = "ddcolor", device=None):
        self.model_name = model_name
        self.input_size = MODEL_CONFIGS[model_name]["input_size"]
        model = build_model(model_name=model_name, device=device)
        self.pipeline = ColorizationPipeline(model, input_size=self.input_size, device=device)

    def filter(
        self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True
    ) -> PilImage:
        # Convert PIL RGB -> OpenCV BGR
        img_rgb = np.asarray(filtered_image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Run DDColor
        result_bgr = self.pipeline.process(img_bgr)

        # Convert back to PIL RGB
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        return PilImage.fromarray(result_rgb)
