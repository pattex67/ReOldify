"""
ReOldify API Server — Colorize images via HTTP.

Usage:
    python -m deoldify.api
    # Then POST an image to http://localhost:8000/colorize
"""

import io
import os
import logging
import tempfile
from pathlib import Path

try:
    from fastapi import FastAPI, File, UploadFile, Query
    from fastapi.responses import StreamingResponse, JSONResponse
except ImportError:
    raise ImportError(
        "FastAPI is required for the API server. Install it with: pip install fastapi uvicorn python-multipart"
    )

from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ReOldify API",
    description="Colorize and restore old images using deep learning",
    version="1.0.0",
)

# Lazy-loaded colorizers
_colorizers = {}


def _get_colorizer(artistic: bool = True):
    key = "artistic" if artistic else "stable"
    if key not in _colorizers:
        from deoldify import device as device_settings
        from deoldify.device_id import DeviceId

        device_env = os.environ.get("REOLDIFY_DEVICE", "cpu")
        if device_env == "cpu":
            device_settings.set(DeviceId.CPU)
        else:
            device_settings.set(DeviceId.GPU0)

        from deoldify.visualize import get_image_colorizer

        logger.info(f"Loading {key} colorizer...")
        _colorizers[key] = get_image_colorizer(artistic=artistic)
        logger.info(f"{key} colorizer loaded.")
    return _colorizers[key]


@app.get("/")
def root():
    return {
        "name": "ReOldify API",
        "version": "1.0.0",
        "endpoints": {
            "POST /colorize": "Colorize an uploaded image",
            "GET /health": "Health check",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/colorize")
async def colorize(
    file: UploadFile = File(..., description="Image file to colorize (JPEG, PNG)"),
    model: str = Query("artistic", enum=["artistic", "stable"], description="Model to use"),
    render_factor: int = Query(35, ge=5, le=50, description="Render quality factor"),
    watermark: bool = Query(False, description="Add watermark to output"),
):
    """Colorize an uploaded image and return the result."""
    # Validate file type
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"error": f"Expected an image file, got: {content_type}"},
        )

    # Save uploaded image to temp file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        artistic = model == "artistic"
        colorizer = _get_colorizer(artistic=artistic)

        result_image = colorizer.get_transformed_image(
            path=tmp_path,
            render_factor=render_factor,
            watermarked=watermark,
        )

        # Convert result to bytes
        img_bytes = io.BytesIO()
        result_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return StreamingResponse(
            img_bytes,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=colorized_{file.filename}"},
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting ReOldify API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
