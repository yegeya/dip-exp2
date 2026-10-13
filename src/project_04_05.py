import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = PROJECT_DIR / "input" / "4.41a.jpg"
DEFAULT_TEMPLATE = PROJECT_DIR / "input" / "4.41b.png"
DEFAULT_RESULTS = PROJECT_DIR / "results" / "project_04_05"
TEMPLATE_CROP = (6, 1, 44, 43)


def load_grayscale(path):
    with Image.open(path) as image:
        return image.convert("L")


def frequency_correlation(image, template, padded_shape):
    image_array = np.asarray(image, dtype=np.float64)
    template_array = np.asarray(template, dtype=np.float64)
    image_height, image_width = image_array.shape
    template_height, template_width = template_array.shape
    padded_height, padded_width = padded_shape

    if padded_height < image_height + template_height - 1:
        raise ValueError("补零后的高度不足，无法计算线性相关")
    if padded_width < image_width + template_width - 1:
        raise ValueError("补零后的宽度不足，无法计算线性相关")

    # 两幅图都放在左上角，其余位置补零，避免循环相关造成的混叠。
    image_padded = np.zeros(padded_shape, dtype=np.float64)
    template_padded = np.zeros(padded_shape, dtype=np.float64)
    image_padded[:image_height, :image_width] = image_array
    template_padded[:template_height, :template_width] = template_array

    # 相关定理：空间域相关 = 图像频谱乘模板频谱的共轭，再做逆变换。
    image_spectrum = np.fft.fft2(image_padded)
    template_spectrum = np.fft.fft2(template_padded)
    correlation = np.fft.ifft2(image_spectrum * np.conj(template_spectrum)).real
    return correlation, image_padded, template_padded


def verify_spatial_correlation(image, template, correlation):
    image_array = np.asarray(image, dtype=np.float64)
    template_array = np.asarray(template, dtype=np.float64)
    template_height, template_width = template_array.shape
    valid_height = image_array.shape[0] - template_height + 1
    valid_width = image_array.shape[1] - template_width + 1
    spatial = np.empty((valid_height, valid_width), dtype=np.float64)

    # 对模板完全位于原图内的每个位置，独立计算空间域乘积和。
    for row in range(valid_height):
        for column in range(valid_width):
            region = image_array[row:row + template_height, column:column + template_width]
            spatial[row, column] = np.sum(region * template_array)

    spatial_peak_y, spatial_peak_x = np.unravel_index(np.argmax(spatial), spatial.shape)
    max_error = float(np.max(np.abs(spatial - correlation[:valid_height, :valid_width])))
    if not np.allclose(spatial, correlation[:valid_height, :valid_width], rtol=0, atol=1e-6):
        raise ValueError(f"空间域与频域相关结果不一致，最大绝对误差为 {max_error}")
    return int(spatial_peak_x), int(spatial_peak_y), float(spatial[spatial_peak_y, spatial_peak_x]), max_error


def display_correlation(correlation, logarithmic=False):
    # 仅为显示映射到 0–255；峰值定位仍使用原始浮点相关数组。
    nonnegative = np.maximum(correlation, 0)
    peak_value = nonnegative.max()
    if peak_value == 0:
        return Image.fromarray(np.zeros(correlation.shape, dtype=np.uint8))
    if logarithmic:
        scaled = np.log1p(nonnegative) / np.log1p(peak_value)
    else:
        scaled = nonnegative / peak_value
    return Image.fromarray(np.round(255 * scaled).astype(np.uint8))


def show_results(image, template, correlation_display, marked, peak_x, peak_y):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Microsoft JhengHei"]
    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    panels = (
        (image, "输入图像"),
        (template, "裁剪后的模板"),
        (correlation_display, "二维相关结果"),
        (marked, f"最大值位置：(x, y) = ({peak_x}, {peak_y})"),
    )
    for axis, (panel, title) in zip(axes.flat, panels):
        axis.imshow(panel, cmap="gray" if panel.mode == "L" else None)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    plt.show()


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="复现 Project 04-05：频率域相关")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="输入图像路径")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE, help="模板图像路径")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="结果保存目录")
    parser.add_argument("--no-show", action="store_true", help="只保存结果，不弹出图像窗口")
    arguments = parser.parse_args()

    image = load_grayscale(arguments.image)
    template_original = load_grayscale(arguments.template)
    if image.size != (256, 256):
        raise ValueError(f"输入图像应为 256×256 像素，实际为 {image.size}")
    if template_original.size != (49, 44):
        raise ValueError(f"裁剪前的模板应为 49×44 像素，实际为 {template_original.size}")

    # Pillow 的裁剪框是 (左, 上, 右, 下)，右、下边界不包含在结果中。
    template = template_original.crop(TEMPLATE_CROP)
    padded_shape = (298, 298)
    correlation, image_padded, template_padded = frequency_correlation(
        image, template, padded_shape
    )

    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    peak_x = int(peak_x)
    peak_y = int(peak_y)
    peak_value = float(correlation[peak_y, peak_x])
    spatial_peak_x, spatial_peak_y, spatial_peak_value, max_error = verify_spatial_correlation(
        image, template, correlation
    )
    if (spatial_peak_x, spatial_peak_y) != (peak_x, peak_y):
        raise ValueError("空间域与频域得到的最大值位置不一致")

    arguments.results.mkdir(parents=True, exist_ok=True)
    correlation_display = display_correlation(correlation)
    template.save(arguments.results / "04_05_template_cropped.png")
    Image.fromarray(image_padded.astype(np.uint8)).save(
        arguments.results / "04_05_image_padded.png"
    )
    Image.fromarray(template_padded.astype(np.uint8)).save(
        arguments.results / "04_05_template_padded.png"
    )
    correlation_display.save(arguments.results / "04_05_correlation_display.png")
    display_correlation(correlation, logarithmic=True).save(
        arguments.results / "04_05_correlation_log_display.png"
    )
    np.save(arguments.results / "04_05_correlation.npy", correlation)

    marked = image.convert("RGB")
    draw = ImageDraw.Draw(marked)
    draw.rectangle(
        (peak_x, peak_y, peak_x + template.width - 1, peak_y + template.height - 1),
        outline="red",
        width=2,
    )
    marked.save(arguments.results / "04_05_match_location.png")

    details = {
        "image": str(arguments.image),
        "image_size_xy": list(image.size),
        "template": str(arguments.template),
        "template_original_size_xy": list(template_original.size),
        "template_crop_xyxy_exclusive": list(TEMPLATE_CROP),
        "template_cropped_size_xy": list(template.size),
        "padded_shape_height_width": list(padded_shape),
        "peak_xy_zero_based": [peak_x, peak_y],
        "peak_value": peak_value,
        "coordinate_meaning": "模板左上角位置；x 为列，y 为行，均从零开始",
        "spatial_peak_xy_zero_based": [spatial_peak_x, spatial_peak_y],
        "spatial_peak_value": spatial_peak_value,
        "spatial_fft_max_absolute_error_valid_region": max_error,
    }
    (arguments.results / "04_05_peak.json").write_bytes(
        (json.dumps(details, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    print(f"相关函数最大值位置：(x, y) = ({peak_x}, {peak_y})；最大值 = {peak_value:.6f}")
    print(f"空间域校验：最大值位置 = ({spatial_peak_x}, {spatial_peak_y})；最大绝对误差 = {max_error:.3e}")
    print(f"结果已保存至：{arguments.results}")
    if not arguments.no_show:
        show_results(image, template, correlation_display, marked, peak_x, peak_y)


if __name__ == "__main__":
    main()
