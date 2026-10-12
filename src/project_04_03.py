"""Project 04-03 Gaussian lowpass filtering, DIP 2/e Eq. (4.3-7)."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import PIL
from common_io import read_gray, validate_image, save_gray, normalize_display, save_panel

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "Fig0441(a)(characters_test_pattern).tif"


def gaussian_lowpass(shape, d0, center=None):
    """Generate H=exp(-D^2/(2*d0^2)); coordinates are (row, column).

    Arbitrary positive integer M,N and finite center coordinates are supported.
    Default fftshift origin is (M//2,N//2), including odd-sized arrays.
    Off-origin responses are for filter construction; they need not give a real
    spatial result. apply_frequency_filter therefore retains complex values.
    """
    if len(shape) != 2 or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n <= 0 for n in shape):
        raise ValueError("shape must contain two positive integers")
    if not np.isfinite(d0) or d0 <= 0:
        raise ValueError("d0 must be finite and positive")
    center = (shape[0] // 2, shape[1] // 2) if center is None else center
    if len(center) != 2 or not np.isfinite(center).all():
        raise ValueError("center must contain two finite coordinates")
    row, col = np.ogrid[:shape[0], :shape[1]]
    return np.exp(-0.5 * (((row - center[0]) / d0) ** 2 + ((col - center[1]) / d0) ** 2))


def apply_frequency_filter(image, response):
    """Return centered input spectrum, filtered spectrum, COMPLEX inverse."""
    image = validate_image(image)
    response = validate_image(response)
    if image.shape != response.shape:
        raise ValueError("image and response shapes must match")
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    filtered = spectrum * response
    inverse = np.fft.ifft2(np.fft.ifftshift(filtered))
    return spectrum, filtered, inverse


def filter_image(image, d0, padding=False):
    """Return unquantized lowpass image, H, F, G, max imaginary residual.

    Default: no padding, periodic boundaries (allowed assignment approximation).
    padding=True: zero-pad at bottom/right to 2M x 2N and crop top-left M x N.
    d0 is in bins of the FFT actually computed, so padding changes its scale.
    """
    image = validate_image(image)
    height, width = image.shape
    padded = np.pad(image, ((0, height), (0, width))) if padding else image
    response = gaussian_lowpass(padded.shape, d0)
    spectrum, filtered, inverse = apply_frequency_filter(padded, response)
    residual = float(np.max(np.abs(inverse.imag)))
    if residual > 1e-10 * max(1.0, float(np.max(np.abs(inverse.real)))):
        raise ValueError("Unexpected imaginary component in centered lowpass result")
    return inverse.real[:height, :width], response, spectrum, filtered, residual


def gradient_energy(image):
    """Mean squared periodic horizontal + vertical first difference."""
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
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--d0", type=float, nargs="+", default=[5,15,30,80,230])
    parser.add_argument("--padding", action="store_true")
    parser.add_argument("--demo-shape", type=int, nargs=2, default=[180,240], metavar=("M","N"))
    parser.add_argument("--demo-center", type=float, nargs=2, default=[60,160], metavar=("ROW","COL"))
    args = parser.parse_args()
    metadata = run(args.input, args.output, args.d0, args.padding, args.demo_shape, args.demo_center)
    print(json.dumps({"shape":metadata["shape"],"d0":metadata["d0_values"],"output":str(args.output)},ensure_ascii=False))
