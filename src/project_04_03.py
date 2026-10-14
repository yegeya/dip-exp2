import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import PIL
from common_io import read_gray, save_gray, normalize_display, save_panel

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "input" / "Fig0441(a)(characters_test_pattern).tif"
def gaussian_lowpass(shape, d0, center=None):
    valid_shape = len(shape) == 2 and all(
        isinstance(size, (int, np.integer)) and size > 0 for size in shape
    )
    if not valid_shape or not np.isfinite(d0) or d0 <= 0:
        raise ValueError("shape and d0 must be positive")
    if center is None:
        center = (shape[0] // 2, shape[1] // 2)
    row, col = np.ogrid[:shape[0], :shape[1]]
    return np.exp(-0.5 * (((row - center[0]) / d0) ** 2 + ((col - center[1]) / d0) ** 2))

def apply_frequency_filter(image, response):
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    filtered = spectrum * response
    inverse = np.fft.ifft2(np.fft.ifftshift(filtered))
    return spectrum, filtered, inverse


def filter_image(image, d0, padding=False):
    height, width = image.shape
    padded = np.pad(image, ((0, height), (0, width))) if padding else image
    response = gaussian_lowpass(padded.shape, d0)
    spectrum, filtered, inverse = apply_frequency_filter(padded, response)
    residual = float(np.max(np.abs(inverse.imag)))
    return inverse.real[:height, :width], response, spectrum, filtered, residual

def gradient_energy(image):
    return float(np.mean((image - np.roll(image, 1, 0)) ** 2 + (image - np.roll(image, 1, 1)) ** 2))

def run(input_path, output_dir, d0_values=(5, 15, 30, 80, 230), padding=False,
        demo_shape=(180, 240), demo_center=(60, 160)):
    image = read_gray(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_gray(image, output_dir / "04_03_original.png")
    F = np.fft.fftshift(np.fft.fft2(np.pad(image, ((0, image.shape[0]), (0, image.shape[1]))) if padding else image))
    log_F = np.log1p(np.abs(F))
    spectrum_limits = (0.0, float(log_F.max()))
    F_display = normalize_display(log_F, spectrum_limits)
    save_gray(F_display, output_dir / "04_03_input_spectrum.png")
    records, responses, spectra, outputs = [], [], [], []
    original_gradient = gradient_energy(image)
    for d0 in d0_values:
        output, H, F, G, residual = filter_image(image, d0, padding)
        label = f"{d0:g}".replace(".", "p")
        save_gray(H, output_dir / f"04_03_filter_d0_{label}.png")
        G_display = normalize_display(np.log1p(np.abs(G)), spectrum_limits)
        save_gray(G_display, output_dir / f"04_03_filtered_spectrum_d0_{label}.png")
        save_gray(output, output_dir / f"04_03_output_d0_{label}.png")
        np.save(output_dir / f"04_03_lowpass_d0_{label}.npy", output)
        records.append({"d0":float(d0), "mean":float(output.mean()), "std":float(output.std()),
                        "rmse_from_original":float(np.sqrt(np.mean((image - output) ** 2))),
                        "gradient_energy":gradient_energy(output),
                        "gradient_retention_percent":100 * gradient_energy(output) / original_gradient if original_gradient else 0.0,
                        "spectral_energy_retention_percent":100 * float(np.sum(np.abs(G) ** 2) / np.sum(np.abs(F) ** 2)) if np.any(F) else 0.0,
                        "imaginary_residual_max":residual})
        responses.append(H); spectra.append(G_display); outputs.append(output)
    labels = [f"({chr(97+i)}) D0 = {d0:g}" for i,d0 in enumerate(d0_values)]
    save_panel(responses, labels, output_dir / "04_03_filters_panel.png", columns=3)
    save_panel(spectra, labels, output_dir / "04_03_spectra_panel.png", columns=3)
    save_panel(outputs, labels, output_dir / "04_03_outputs_panel.png", columns=3)
    save_panel([image] + outputs, ["(a) Original"] + [f"({chr(98+i)}) D0 = {d:g}" for i,d in enumerate(d0_values)], output_dir / "04_03_comparison.png", columns=2)
    demo = gaussian_lowpass(tuple(demo_shape), 15, tuple(demo_center))
    save_gray(demo, output_dir / "04_03_custom_center_filter.png")
    with (output_dir / "04_03_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    metadata = {"input_filename":Path(input_path).name,
                "input_sha256":hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
                "shape":list(image.shape), "fft_shape":list(F.shape), "padding":padding,
                "center":[F.shape[0]//2,F.shape[1]//2], "d0_values":list(d0_values),
                "original_mean":float(image.mean()),"original_std":float(image.std()),
                "original_gradient_energy":original_gradient,"spectrum_display_limits":list(spectrum_limits),
                "custom_filter_shape":list(demo_shape),"custom_filter_center":list(demo_center),
                "environment":{"os":platform.platform(),"python":platform.python_version(),"numpy":np.__version__,"pillow":PIL.__version__},
                "metrics":records}
    (output_dir / "04_03_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "project_04_03")
    parser.add_argument("--d0", type=float, nargs="+", default=[5,15,30,80,230])
    parser.add_argument("--padding", action="store_true")
    parser.add_argument("--demo-shape", type=int, nargs=2, default=[180,240], metavar=("M","N"))
    parser.add_argument("--demo-center", type=float, nargs=2, default=[60,160], metavar=("ROW","COL"))
    args = parser.parse_args()
    metadata = run(args.input, args.output, args.d0, args.padding, args.demo_shape, args.demo_center)
    print(json.dumps({"shape":metadata["shape"],"d0":metadata["d0_values"],"output":str(args.output)},ensure_ascii=False))
