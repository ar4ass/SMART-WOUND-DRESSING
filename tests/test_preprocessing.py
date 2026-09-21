import io

import numpy as np
import pytest
from PIL import Image

from src.preprocessing import (
    ImageLoadError,
    STANDARD_MAX_WIDTH_PX,
    denoise_light,
    load_image_from_bytes,
    preprocess_image,
    resize_to_standard_width,
)


def _make_fake_jpeg_bytes(width=100, height=80, color=(200, 50, 50)) -> bytes:
    image = Image.new("RGB", (width, height), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_valid_image_loads_correctly():
    file_bytes = _make_fake_jpeg_bytes(120, 90)
    result = load_image_from_bytes(file_bytes, file_name="dummy.jpg")

    assert result.width == 120
    assert result.height == 90
    assert result.color_mode == "RGB"
    assert isinstance(result.array, np.ndarray)
    assert result.array.shape == (90, 120, 3)
    assert result.size_kb > 0


def test_empty_bytes_raises_error():
    with pytest.raises(ImageLoadError):
        load_image_from_bytes(b"", file_name="empty.jpg")


def test_non_image_bytes_raises_error():
    with pytest.raises(ImageLoadError):
        load_image_from_bytes(b"ini bukan gambar sama sekali", file_name="fake.jpg")


def test_too_small_image_raises_error():
    tiny_bytes = _make_fake_jpeg_bytes(10, 10)
    with pytest.raises(ImageLoadError):
        load_image_from_bytes(tiny_bytes, file_name="tiny.jpg")


def test_rgba_png_is_converted_to_rgb():
    image = Image.new("RGBA", (100, 100), color=(10, 20, 30, 128))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    result = load_image_from_bytes(buffer.getvalue(), file_name="alpha.png")

    assert result.color_mode == "RGB"
    assert result.array.shape == (100, 100, 3)


def test_resize_shrinks_wide_image_and_keeps_aspect_ratio():
    array = np.zeros((800, 1600, 3), dtype=np.uint8)

    resized = resize_to_standard_width(array, max_width=STANDARD_MAX_WIDTH_PX)

    assert resized.shape[1] == STANDARD_MAX_WIDTH_PX
    assert resized.shape[0] == STANDARD_MAX_WIDTH_PX // 2


def test_resize_does_not_enlarge_small_image():
    array = np.zeros((100, 200, 3), dtype=np.uint8)

    resized = resize_to_standard_width(array, max_width=STANDARD_MAX_WIDTH_PX)

    assert resized.shape == array.shape


def test_denoise_keeps_shape_and_dtype():
    array = (np.random.rand(50, 60, 3) * 255).astype(np.uint8)

    denoised = denoise_light(array)

    assert denoised.shape == array.shape
    assert denoised.dtype == array.dtype


def test_preprocess_image_runs_resize_then_denoise():
    array = np.full((400, 1200, 3), 120, dtype=np.uint8)

    result = preprocess_image(array)

    assert result.shape[1] == STANDARD_MAX_WIDTH_PX
    assert result.dtype == np.uint8
