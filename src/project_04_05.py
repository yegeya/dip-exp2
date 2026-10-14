import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# 输入输出的相对路径设置
ROOT_DIR = Path(__file__).resolve().parents[1]
IMG_PATH = ROOT_DIR / "input" / "4.41a.jpg"
TPL_PATH = ROOT_DIR / "input" / "4.41b.png"
OUT_DIR = ROOT_DIR / "results" / "project_04_05"
CROP_BOX = (6, 1, 44, 43)
# 不需要弹窗时改为 False。
SHOW_RESULTS = True


def load_grayscale(path):
    return Image.open(path).convert("L")


def frequency_correlation(image, template, padded_shape):
    image_array = np.asarray(image)
    template_array = np.asarray(template)
    image_height, image_width = image_array.shape
    template_height, template_width = template_array.shape

    # 图像和模板都放在零数组的左上角
    image_padded = np.zeros(padded_shape)
    template_padded = np.zeros(padded_shape)
    image_padded[:image_height, :image_width] = image_array
    template_padded[:template_height, :template_width] = template_array

    # 模板频谱取共轭相乘，再逆变换得到相关函数
    image_spectrum = np.fft.fft2(image_padded)
    template_spectrum = np.fft.fft2(template_padded)

    # 频域相关函数
    correlation = np.fft.ifft2(image_spectrum * np.conj(template_spectrum)).real
    return correlation, image_padded, template_padded


# 验证频域相关函数与空间相关函数的一致性
def verify_spatial_correlation(image, template, correlation):
    image_array = np.asarray(image, dtype=float)
    template_array = np.asarray(template, dtype=float)
    template_height, template_width = template_array.shape
    valid_height = image_array.shape[0] - template_height + 1
    valid_width = image_array.shape[1] - template_width + 1
    spatial = np.empty((valid_height, valid_width))

    # 逐位置计算像素乘积和，与频域结果比较。
    for row in range(valid_height):
        for column in range(valid_width):
            region = image_array[row:row + template_height, column:column + template_width]
            spatial[row, column] = np.sum(region * template_array)

    spatial_peak_y, spatial_peak_x = np.unravel_index(np.argmax(spatial), spatial.shape)
    max_error = float(np.max(np.abs(spatial - correlation[:valid_height, :valid_width])))
    return int(spatial_peak_x), int(spatial_peak_y), float(spatial[spatial_peak_y, spatial_peak_x]), max_error


def display_correlation(correlation, logarithmic=False):
    # 显示用的灰度图，找最大值时仍使用原始相关数组
    nonnegative = np.maximum(correlation, 0)
    peak_value = nonnegative.max()
    if logarithmic:
        scaled = np.log1p(nonnegative) / np.log1p(peak_value)
    else:
        scaled = nonnegative / peak_value
    return Image.fromarray(np.round(255 * scaled).astype(np.uint8))


# 展示结果
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
    image = load_grayscale(IMG_PATH)
    template_original = load_grayscale(TPL_PATH)

    # 裁剪框是 (左, 上, 右, 下)，右、下边界不包含在内，模板与示例大小不一样所以进行裁剪
    template = template_original.crop(CROP_BOX)
    padded_shape = (298, 298)
    correlation, image_padded, template_padded = frequency_correlation(image, template, padded_shape)

    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    peak_x = int(peak_x)
    peak_y = int(peak_y)
    peak_value = float(correlation[peak_y, peak_x])
    spatial_peak_x, spatial_peak_y, spatial_peak_value, max_error = verify_spatial_correlation(
        image, template, correlation
    )

    # 保存结果
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    correlation_display = display_correlation(correlation)
    template.save(OUT_DIR / "04_05_template_cropped.png")
    Image.fromarray(image_padded.astype(np.uint8)).save(
        OUT_DIR / "04_05_image_padded.png"
    )
    Image.fromarray(template_padded.astype(np.uint8)).save(
        OUT_DIR / "04_05_template_padded.png"
    )
    correlation_display.save(OUT_DIR / "04_05_correlation_display.png")
    display_correlation(correlation, logarithmic=True).save(
        OUT_DIR / "04_05_correlation_log_display.png"
    )
    np.save(OUT_DIR / "04_05_correlation.npy", correlation)

    marked = image.convert("RGB")
    draw = ImageDraw.Draw(marked)
    draw.rectangle(
        (peak_x, peak_y, peak_x + template.width - 1, peak_y + template.height - 1),
        outline="red",
        width=2,
    )
    marked.save(OUT_DIR / "04_05_match_location.png")

    result_info = {
        "peak_xy_zero_based": [peak_x, peak_y],
        "peak_value": peak_value,
        "spatial_peak_xy_zero_based": [spatial_peak_x, spatial_peak_y],
        "spatial_peak_value": spatial_peak_value,
        "spatial_fft_max_absolute_error_valid_region": max_error,
    }
    (OUT_DIR / "04_05_peak.json").write_text(
        json.dumps(result_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"相关函数最大值位置：(x, y) = ({peak_x}, {peak_y})；最大值 = {peak_value:.6f}")
    print(f"空间域校验：最大值位置 = ({spatial_peak_x}, {spatial_peak_y})；最大绝对误差 = {max_error:.3e}")
    print(f"结果已保存至：{OUT_DIR}")
    if SHOW_RESULTS:
        show_results(image, template, correlation_display, marked, peak_x, peak_y)


if __name__ == "__main__":
    main()
