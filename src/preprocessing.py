from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

MIN_WIDTH_PX = 50
MIN_HEIGHT_PX = 50

STANDARD_MAX_WIDTH_PX = 800


class ImageLoadError(Exception):
    pass


@dataclass
class LoadedImage:

    image: Image.Image
    array: np.ndarray
    file_name: str
    size_kb: float
    width: int
    height: int
    color_mode: str
    format: str | None


def load_image_from_bytes(file_bytes: bytes, file_name: str = "uploaded") -> LoadedImage:
    if file_bytes is None or len(file_bytes) == 0:
        raise ImageLoadError("File kosong (0 byte). Silakan unggah file gambar yang valid.")

    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()
    except UnidentifiedImageError as exc:
        raise ImageLoadError(
            "File tidak dikenali sebagai gambar. Pastikan formatnya JPG atau PNG."
        ) from exc
    except OSError as exc:
        raise ImageLoadError(
            "Gambar tampak rusak atau tidak lengkap (gagal dibaca)."
        ) from exc

    width, height = image.size
    if width < MIN_WIDTH_PX or height < MIN_HEIGHT_PX:
        raise ImageLoadError(
            f"Gambar terlalu kecil ({width}x{height}px). "
            f"Minimal {MIN_WIDTH_PX}x{MIN_HEIGHT_PX}px."
        )

    if image.mode != "RGB":
        image = image.convert("RGB")

    array = np.array(image)

    return LoadedImage(
        image=image,
        array=array,
        file_name=file_name,
        size_kb=round(len(file_bytes) / 1024, 1),
        width=width,
        height=height,
        color_mode=image.mode,
        format=image.format,
    )


def resize_to_standard_width(
    array: np.ndarray, max_width: int = STANDARD_MAX_WIDTH_PX
) -> np.ndarray:
    height, width = array.shape[:2]

    if width <= max_width:
        return array

    scale = max_width / width
    new_width = max_width
    new_height = int(round(height * scale))

    resized = cv2.resize(array, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized


def denoise_light(array: np.ndarray) -> np.ndarray:
    denoised = cv2.GaussianBlur(array, ksize=(5, 5), sigmaX=0)
    return denoised


def preprocess_image(array: np.ndarray) -> np.ndarray:
    resized = resize_to_standard_width(array)
    denoised = denoise_light(resized)
    return denoised
