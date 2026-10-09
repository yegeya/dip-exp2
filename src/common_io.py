"""Shared grayscale image I/O. Processing arrays are float64 in [0, 1]."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def validate_image(image):
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape or not np.isfinite(array).all():
        raise ValueError("Expected a nonempty finite 2D grayscale array")
    return array


def read_gray(path):
    """Read an 8/16-bit grayscale or RGB image; never stretch contrast."""
    with Image.open(path) as image:
        if image.mode in ("RGB", "RGBA", "P", "CMYK", "LA"):
            image = image.convert("L")
        array = np.asarray(image)
        if array.dtype == np.bool_:
            return array.astype(np.float64)
        if array.dtype.kind == "u":
            result = array.astype(np.float64) / np.iinfo(array.dtype).max
        elif array.dtype.kind == "f":
            result = array.astype(np.float64)
        else:
            raise ValueError("Convert signed-integer images to uint8/uint16 first")
    result = validate_image(result)
    if result.min() < 0 or result.max() > 1:
        raise ValueError("Floating image pixels must lie in [0, 1]")
    return result


def normalize_display(array, limits=None):
    """Min-max normalization ONLY for diagnostic displays such as spectra."""
    array = validate_image(array)
    lo, hi = (float(array.min()), float(array.max())) if limits is None else limits
    if hi < lo or not np.isfinite([lo, hi]).all():
        raise ValueError("Invalid display limits")
    return np.zeros_like(array) if hi == lo else np.clip((array - lo) / (hi - lo), 0, 1)


def gray_pil(image):
    """Clip to [0,1], round to 8-bit; no per-image contrast stretching."""
    return Image.fromarray(np.rint(np.clip(validate_image(image), 0, 1) * 255).astype("uint8"))


def save_gray(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gray_pil(image).save(path)
    return path


def show_gray(image):
    """Optional local viewer; experiment CLI never opens windows automatically."""
    gray_pil(image).show()


def save_panel(images, labels, path, columns=3, cell=344):
    """Compose labeled diagnostics without altering the processing arrays."""
    if len(images) != len(labels) or not images or columns < 1:
        raise ValueError("Nonempty images and labels must have the same length")
    rows = (len(images) + columns - 1) // columns
    margin, label_height = 16, 36
    canvas = Image.new("RGB", (margin + columns * (cell + margin), margin + rows * (cell + label_height + margin)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 19)
    except OSError:
        font = ImageFont.load_default(size=19)
    for index, (array, label) in enumerate(zip(images, labels)):
        x = margin + (index % columns) * (cell + margin)
        y = margin + (index // columns) * (cell + label_height + margin)
        tile = gray_pil(array)
        tile.thumbnail((cell, cell), Image.Resampling.LANCZOS)
        canvas.paste(tile, (x + (cell - tile.width) // 2, y + (cell - tile.height) // 2))
        draw.text((x, y + cell + 6), label, fill="black", font=font)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
