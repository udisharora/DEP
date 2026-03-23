import os
from functools import lru_cache

from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_NAFNET_MODEL_DIR = os.path.join(PROJECT_ROOT, "NAFNet", "experiments", "pretrained_models")
DEFAULT_DEBLUR_MODEL_ID = "reds_width64"


@lru_cache(maxsize=None)
def _get_deblur_processor(
    model_id=DEFAULT_DEBLUR_MODEL_ID,
    model_dir=DEFAULT_NAFNET_MODEL_DIR,
    device="auto",
):
    """
    Lazily construct a nafnetlib DeblurProcessor for repeated use.
    """
    try:
        from nafnetlib import DeblurProcessor
    except ImportError as exc:
        raise ImportError(
            "nafnetlib is not installed. Install it in the active environment "
            "before using NAFNet deblurring."
        ) from exc

    resolved_device = device
    if device == "auto":
        try:
            import torch
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            resolved_device = "cpu"

    return DeblurProcessor(
        model_id=model_id,
        model_dir=model_dir,
        device=resolved_device,
    )


def process_with_nafnet(
    image,
    model_id=DEFAULT_DEBLUR_MODEL_ID,
    model_dir=DEFAULT_NAFNET_MODEL_DIR,
    device="auto",
):
    """
    Deblur a single image with nafnetlib and return the result as a PIL image.
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    processor = _get_deblur_processor(
        model_id=model_id,
        model_dir=model_dir,
        device=device,
    )
    return processor.process(image.convert("RGB"))


def nafnet_is_available():
    try:
        from nafnetlib import DeblurProcessor  # noqa: F401
        return True
    except Exception:
        return False
