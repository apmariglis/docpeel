"""
Tests for image_utils.py — base64 encoding, quadrant splitting, obfuscation,
and adaptive JPEG size-capping. No API calls or real PDFs required.
"""

import base64
import io

import pytest
from PIL import Image

from src.docpeel.image_utils import (
    _ANTHROPIC_MAX_B64_BYTES,
    _encode_jpeg,
    obfuscate,
    split_quadrants,
    to_b64,
    to_b64_safe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid(width: int, height: int, color=(100, 150, 200)) -> Image.Image:
    """Return a solid-colour RGB image of the given size."""
    return Image.new("RGB", (width, height), color)


def _decode_b64(s: str) -> bytes:
    return base64.b64decode(s)


# ---------------------------------------------------------------------------
# to_b64
# ---------------------------------------------------------------------------


def test_to_b64_returns_valid_base64():
    img = _solid(10, 10)
    b64 = to_b64(img)
    # Must not raise
    raw = _decode_b64(b64)
    assert len(raw) > 0


def test_to_b64_encodes_as_png():
    img = _solid(10, 10)
    b64 = to_b64(img)
    raw = _decode_b64(b64)
    # PNG magic bytes: \x89PNG
    assert raw[:4] == b"\x89PNG"


def test_to_b64_preserves_dimensions():
    img = _solid(30, 20)
    b64 = to_b64(img)
    raw = _decode_b64(b64)
    decoded = Image.open(io.BytesIO(raw))
    assert decoded.size == (30, 20)


# ---------------------------------------------------------------------------
# split_quadrants
# ---------------------------------------------------------------------------


def test_split_quadrants_returns_four_keys():
    img = _solid(100, 80)
    quads = split_quadrants(img)
    assert set(quads.keys()) == {"top-left", "top-right", "bottom-left", "bottom-right"}


def test_split_quadrants_even_dimensions():
    img = _solid(100, 80)
    quads = split_quadrants(img)
    assert quads["top-left"].size == (50, 40)
    assert quads["top-right"].size == (50, 40)
    assert quads["bottom-left"].size == (50, 40)
    assert quads["bottom-right"].size == (50, 40)


def test_split_quadrants_odd_width():
    img = _solid(101, 80)
    quads = split_quadrants(img)
    # mx = 101 // 2 = 50; right gets the remainder
    assert quads["top-left"].width == 50
    assert quads["top-right"].width == 51


def test_split_quadrants_odd_height():
    img = _solid(100, 81)
    quads = split_quadrants(img)
    # my = 81 // 2 = 40; bottom gets the remainder
    assert quads["top-left"].height == 40
    assert quads["bottom-left"].height == 41


def test_split_quadrants_cover_full_area():
    """Sum of quadrant pixel counts equals original image pixel count."""
    img = _solid(100, 80)
    quads = split_quadrants(img)
    total = sum(q.width * q.height for q in quads.values())
    assert total == 100 * 80


# ---------------------------------------------------------------------------
# obfuscate
# ---------------------------------------------------------------------------


def test_obfuscate_returns_pil_image():
    img = _solid(50, 50)
    result = obfuscate(img)
    assert isinstance(result, Image.Image)


def test_obfuscate_expands_size_due_to_rotation():
    img = _solid(100, 100)
    result = obfuscate(img)
    # A 1.5° rotation with expand=True must increase at least one dimension
    assert result.width >= img.width and result.height >= img.height


def test_obfuscate_changes_pixel_values():
    """Noise means the output is bitwise different from the input."""
    img = _solid(50, 50, color=(128, 128, 128))
    original_bytes = img.tobytes()
    result = obfuscate(img)
    # Crop back to original size to compare same region
    cropped = result.crop((0, 0, img.width, img.height))
    assert cropped.tobytes() != original_bytes


def test_obfuscate_pixel_values_in_range():
    """No pixel channel should be outside 0–255 after noise."""
    import numpy as np

    img = _solid(30, 30, color=(0, 0, 0))  # dark corner case
    result = obfuscate(img)
    arr = np.array(result)
    assert arr.min() >= 0
    assert arr.max() <= 255


# ---------------------------------------------------------------------------
# _encode_jpeg
# ---------------------------------------------------------------------------


def test_encode_jpeg_returns_valid_base64():
    img = _solid(50, 50)
    b64 = _encode_jpeg(img)
    raw = _decode_b64(b64)
    assert raw[:2] == b"\xff\xd8"  # JPEG SOI marker


def test_encode_jpeg_drops_alpha():
    """RGBA input should be accepted (alpha stripped silently)."""
    img = Image.new("RGBA", (20, 20), (255, 0, 0, 128))
    b64 = _encode_jpeg(img)
    raw = _decode_b64(b64)
    assert raw[:2] == b"\xff\xd8"


def test_encode_jpeg_smaller_than_png():
    img = _solid(200, 200)
    png_b64 = to_b64(img)
    jpeg_b64 = _encode_jpeg(img)
    assert len(jpeg_b64) < len(png_b64)


# ---------------------------------------------------------------------------
# to_b64_safe
# ---------------------------------------------------------------------------


def test_to_b64_safe_returns_valid_base64():
    img = _solid(100, 100)
    b64 = to_b64_safe(img)
    raw = _decode_b64(b64)
    assert len(raw) > 0


def test_to_b64_safe_normal_image_within_limit():
    img = _solid(200, 200)
    b64 = to_b64_safe(img)
    assert len(b64) <= _ANTHROPIC_MAX_B64_BYTES


def test_to_b64_safe_downscales_oversized():
    """With a tiny max_bytes limit, the function must downscale."""
    img = _solid(500, 500)
    tiny_limit = 1000  # bytes — much smaller than any real image
    b64 = to_b64_safe(img, max_bytes=tiny_limit)
    # Must still return valid base64 (safety floor kicks in)
    raw = _decode_b64(b64)
    assert raw[:2] == b"\xff\xd8"


def test_to_b64_safe_result_is_jpeg():
    img = _solid(100, 100)
    b64 = to_b64_safe(img)
    raw = _decode_b64(b64)
    assert raw[:2] == b"\xff\xd8"
