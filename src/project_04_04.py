from pathlib import Path

import numpy as np
from common_io import read_gray
from project_04_03 import filter_image
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = PROJECT_ROOT / "input" / "Fig0441(a)(characters_test_pattern).tif"
RESULTS = PROJECT_ROOT / "results" / "project_04_04"
RESULTS.mkdir(parents=True, exist_ok=True)
D0_VALUES = [5, 15, 30, 80, 230]
REFERENCE_GHPF_D0_VALUES = [15, 30, 80]
REFERENCE_GHPF_ORDER = 2


def highpass_to_u8(highpass):
    return (np.clip(highpass, 0, 1) * 255).astype(np.uint8)


def save_comparison_panel(images, labels, path, columns=3):
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")
    tiles = [highpass_to_u8(image) for image in images]
    height, width = tiles[0].shape
    label_height = 36
    rows = (len(tiles) + columns - 1) // columns
    panel = np.full((rows * (height + label_height), columns * width), 255, dtype=np.uint8)
    for index, (tile, label) in enumerate(zip(tiles, labels)):
        row = index // columns
        col = index % columns
        y = row * (height + label_height)
        x = col * width
        panel[y:y + height, x:x + width] = tile
        cv2.putText(
            panel,
            label,
            (x + 12, y + height + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            0,
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(path, panel)


def save_difference_image(image, reference, path, scale=4):
    diff = np.abs(highpass_to_u8(image).astype(np.int16) - highpass_to_u8(reference).astype(np.int16))
    cv2.imwrite(path, np.clip(diff * scale, 0, 255).astype(np.uint8))


def gaussian_highpass(shape, d0, order=2):
    # 按广义高斯高通公式构造中心化的 n 阶频域滤波器。
    if d0 <= 0:
        raise ValueError("d0 must be positive")
    if order <= 0:
        raise ValueError("order must be positive")
    center = (shape[0] // 2, shape[1] // 2)
    row, col = np.ogrid[:shape[0], :shape[1]]
    distance2 = (row - center[0]) ** 2 + (col - center[1]) ** 2
    return 1 - np.exp(-0.5 * (distance2 / (d0 ** 2)) ** order)


def apply_frequency_response(image, response):
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    filtered = spectrum * response
    inverse = np.fft.ifft2(np.fft.ifftshift(filtered))
    return inverse.real


def save_order_response_comparison(path, max_normalized_frequency=4.0, samples=1200):
    # 绘制一阶和二阶广义高斯高通滤波器的归一化响应，并以 dB 形式比较其低频抑制和高频通过特性。
    # 横轴使用 D / D0，用于比较滤波器阶数
    normalized_frequency = np.linspace(0, max_normalized_frequency, samples)

    first_order = 1 - np.exp(-0.5 * normalized_frequency ** 2)
    second_order = 1 - np.exp(-0.5 * normalized_frequency ** 4)

    epsilon = 1e-12
    first_order_db = 20 * np.log10(np.maximum(first_order, epsilon))
    second_order_db = 20 * np.log10(np.maximum(second_order, epsilon))

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    axes[0].plot(normalized_frequency, first_order, label="order 1: 1 - H_LP", linewidth=2)
    axes[0].plot(normalized_frequency, second_order, label="order 2: GHPF", linewidth=2)
    axes[0].axvline(1.0, color="0.35", linestyle="--", linewidth=1, label="D / D0 = 1")
    axes[0].set_title("Normalized Gaussian high-pass response")
    axes[0].set_xlabel("Normalized frequency D / D0")
    axes[0].set_ylabel("Amplitude response H_HP")
    axes[0].set_xlim(0, max_normalized_frequency)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].plot(normalized_frequency, first_order_db, label="order 1", linewidth=2)
    axes[1].plot(normalized_frequency, second_order_db, label="order 2", linewidth=2)
    axes[1].axvline(1.0, color="0.35", linestyle="--", linewidth=1, label="D / D0 = 1")
    axes[1].set_title("Gaussian high-pass response in dB")
    axes[1].set_xlabel("Normalized frequency D / D0")
    axes[1].set_ylabel("20 log10(|H_HP|) (dB)")
    axes[1].set_xlim(0, max_normalized_frequency)
    axes[1].set_ylim(-60, 1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    figure.savefig(path, dpi=180)
    plt.close(figure)

# 读取原图
original = read_gray(INPUT)

# 绘制一阶等效高通与二阶 GHPF 的径向频率响应对比图。
save_order_response_comparison(
    RESULTS / "04_04_gaussian_highpass_order_comparison.png"
)
# 04-03实验中得到当D0=15时，处理结果最接近Fig4.18(c)中的图像，复用04-03中的低通滤波函数
lowpass, *_ = filter_image(original, d0=15)
# partA
# 直接使用原图像减去04-03(b)得到的低通图
highpass = original - lowpass
part_a_highpass = highpass.copy()

# 保存真实高通数据
np.save(RESULTS / "04_04_part_a_highpass_d0_15.npy", highpass)

# 保存显示图
cv2.imwrite(str(RESULTS / "04_04_part_a_highpass_d0_15.png"), highpass_to_u8(highpass))

# partB
# 做不同参数的对比实验，比较不同的参数下，哪个结果最接近Fig.4.26(c)
comparison_images = []
comparison_labels = []

for d0 in D0_VALUES:
    lowpass, *_ = filter_image(original, d0=d0)
    highpass = original - lowpass

    label = f"{d0:g}"

    np.save(RESULTS / f"04_04_highpass_d0_{label}.npy", highpass)
    cv2.imwrite(str(RESULTS / f"04_04_highpass_d0_{label}.png"), highpass_to_u8(highpass))

    comparison_images.append(highpass)
    comparison_labels.append(f"D0 = {label}")

save_comparison_panel(
    comparison_images,
    comparison_labels,
    str(RESULTS / "04_04_highpass_comparison.png"),
)

# 上述部分结果发现80是最接近的参数，但并未和教材给出的图十分相似，接下来进行细调
D0_VALUES_2 = [60, 65, 70, 75, 80, 85, 90]
comparison_images_2 = []
comparison_labels_2 = []

for d0 in D0_VALUES_2:
    lowpass, *_ = filter_image(original, d0=d0)
    highpass = original - lowpass

    label = f"{d0:g}"

    np.save(RESULTS / f"04_04_highpass_d0_{label}.npy", highpass)
    cv2.imwrite(str(RESULTS / f"04_04_highpass_d0_{label}.png"), highpass_to_u8(highpass))

    comparison_images_2.append(highpass)
    comparison_labels_2.append(f"D0 = {label}")

save_comparison_panel(
    comparison_images_2,
    comparison_labels_2,
    str(RESULTS / "04_04_highpass_comparison_2.png"),
)

# 上述部分结果发现70是最接近的参数，再次进行细调查看是否有更接近的参数
D0_VALUES_3 = list(range(65, 76))
comparison_images_3 = []
comparison_labels_3 = []

for d0 in D0_VALUES_3:
    lowpass, *_ = filter_image(original, d0=d0)
    highpass = original - lowpass

    label = f"{d0:g}"

    np.save(RESULTS / f"04_04_highpass_d0_{label}.npy", highpass)
    cv2.imwrite(str(RESULTS / f"04_04_highpass_d0_{label}.png"), highpass_to_u8(highpass))

    comparison_images_3.append(highpass)
    comparison_labels_3.append(f"D0 = {label}")

save_comparison_panel(
    comparison_images_3,
    comparison_labels_3,
    str(RESULTS / "04_04_highpass_comparison_3.png"),
)

# 生成教材 Fig. 4.26 中指定的二阶 GHPF 对比图。
# 这部分不参与“原图减低通图”的主实验计算，只用于生成 D0=15、30、80 的直接高斯高通滤波参考结果，
# 方便和本实验通过式 (4.4-14) 得到的高通图像进行视觉和误差对比。
reference_images = []
reference_labels = []
reference_d0_15 = None
reference_d0_80 = None

for d0 in REFERENCE_GHPF_D0_VALUES:
    response = gaussian_highpass(original.shape, d0, REFERENCE_GHPF_ORDER)
    highpass = apply_frequency_response(original, response)
    label = f"d0_{d0:g}_order_{REFERENCE_GHPF_ORDER:g}"

    np.save(RESULTS / f"temp_ghpf_{label}.npy", highpass)
    cv2.imwrite(str(RESULTS / f"temp_ghpf_{label}.png"), highpass_to_u8(highpass))

    reference_images.append(highpass)
    reference_labels.append(f"D0 = {d0:g}")
    if d0 == 15:
        reference_d0_15 = highpass.copy()
    if d0 == 80:
        reference_d0_80 = highpass.copy()

save_comparison_panel(
    reference_images,
    reference_labels,
    str(RESULTS / f"temp_ghpf_order_{REFERENCE_GHPF_ORDER:g}_comparison.png"),
)

# Part A 对比图：左侧为式 (4.4-14) 的原图减低通结果，右侧为教材指定的二阶 GHPF D0=15 直接滤波结果。
if reference_d0_15 is not None:
    save_comparison_panel(
        [part_a_highpass, reference_d0_15],
        ["subtract LP D0 = 15", "order-2 GHPF D0 = 15"],
        str(RESULTS / "04_04_part_a_vs_order2_ghpf_d0_15.png"),
        columns=2,
    )

# 生成最佳相减结果 D0=70 与二阶 GHPF D0=80 参考图之间的差异放大图。
if reference_d0_80 is not None:
    best_lowpass, *_ = filter_image(original, d0=70)
    best_highpass = original - best_lowpass
    save_difference_image(
        best_highpass,
        reference_d0_80,
        str(RESULTS / "compare_best_latest_04_04_highpass_d0_70_vs_temp_ghpf_d0_80_order_2_diff_x4.png"),
        scale=4,
    )
