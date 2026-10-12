"""Numerical checks against independent analytic invariants and convolution."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common_io import read_gray, save_gray
from project_04_03 import gaussian_lowpass, filter_image, apply_frequency_filter


class GaussianTests(unittest.TestCase):
    def test_custom_size_center_and_radius(self):
        H = gaussian_lowpass((13, 19), 3, (4, 7))
        self.assertEqual(H.shape, (13, 19))
        self.assertEqual(H[4, 7], 1.0)
        self.assertAlmostEqual(H[4, 10], np.exp(-0.5))
        self.assertAlmostEqual(H[7, 7], H[4, 10])

    def test_dc_and_mean_preserved_even_and_odd(self):
        rng = np.random.default_rng(2026)
        for shape in [(12, 14), (11, 17), (12, 17)]:
            image = rng.random(shape)
            output, _, _, _, residual = filter_image(image, 2)
            self.assertAlmostEqual(output.mean(), image.mean(), places=13)
            self.assertLess(residual, 1e-13)
            uniform = np.full(shape, 0.37)
            self.assertTrue(np.allclose(filter_image(uniform, 2)[0], uniform))

    def test_known_sinusoidal_gain(self):
        rows, cols = np.indices((16, 20))
        image = 0.5 + 0.2 * np.cos(2*np.pi*(3*rows/16 + 4*cols/20))
        expected = 0.5 + (image - 0.5) * np.exp(-25/(2*3**2))
        self.assertTrue(np.allclose(filter_image(image, 3)[0], expected, atol=1e-13))

    def test_frequency_result_matches_direct_circular_convolution(self):
        shape = (5, 7)
        image = np.random.default_rng(10).random(shape)
        H = gaussian_lowpass(shape, 1.2)
        kernel = np.fft.ifft2(np.fft.ifftshift(H)).real
        expected = np.zeros(shape)
        for row in range(shape[0]):
            for col in range(shape[1]):
                expected += kernel[row, col] * np.roll(image, (row, col), (0, 1))
        self.assertTrue(np.allclose(filter_image(image, 1.2)[0], expected, atol=1e-13))

    def test_off_center_keeps_complex_information(self):
        image = np.zeros((16, 16)); image[0, 0] = 1
        _, _, inverse = apply_frequency_filter(image, gaussian_lowpass(image.shape, 2, (4, 7)))
        self.assertTrue(np.iscomplexobj(inverse))
        self.assertGreater(np.abs(inverse.imag).max(), 1e-4)

    def test_io_fixed_gray_scale_and_rgb(self):
        temp_root = Path(__file__).resolve().parent / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            path = Path(directory) / "image.png"
            source = np.array([[0.2,0.4],[0.6,0.8]])
            save_gray(source, path)
            self.assertTrue(np.allclose(source, read_gray(path)))
            Image.new("RGB", (2, 3), (255, 0, 0)).save(path)
            result = read_gray(path)
            self.assertEqual(result.shape, (3, 2))
            self.assertAlmostEqual(result[0, 0], 76/255)

    def test_bad_parameters_and_padded_shape(self):
        for shape, d0 in [((0, 4), 2), ((4, 4), 0), ((4.5, 4), 2), ((4, 4), float("nan"))]:
            with self.assertRaises(ValueError): gaussian_lowpass(shape, d0)
        output, H, *_ = filter_image(np.ones((5, 7)), 2, padding=True)
        self.assertEqual(output.shape, (5, 7))
        self.assertEqual(H.shape, (10, 14))


if __name__ == "__main__": unittest.main()
