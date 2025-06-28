import logging
import io
import os
from typing import Any

import numpy as np
import torch

from data.augmentation import build_inference_transforms
from model import UNet

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def model_fn(model_dir: str) -> torch.nn.Module:
    model_path = os.path.join(model_dir, "best-model.ckpt")
    model = UNet(architecture="fpn")
    model = model.load_from_checkpoint(model_path)
    logger.info("Model loaded")

    model.eval()
    model.freeze()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


def input_fn(request_body: Any, request_content_type: str) -> torch.Tensor:
    if request_content_type == "application/x-npy":
        images = np.load(io.BytesIO(request_body))
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

    if images.ndim == 3:
        images = images[np.newaxis, ...]  

    transforms = build_inference_transforms(target_size=256)
    image_batch = torch.tensor(
        np.stack([transforms(image=img)["image"] for img in images], axis=0)
    )
    return image_batch


def predict_fn(input_data: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    with torch.no_grad():
        logits = model(input_data.to(model.device))
        preds = torch.sigmoid(logits)
        test_masks_prediction = (preds > 0.5).float()
    return test_masks_prediction
