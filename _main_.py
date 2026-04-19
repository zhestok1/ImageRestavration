import numpy as np
from PIL import Image
from skimage import data
from skimage.transform import resize

from hadamard_basis import HadamardBasis
from gauss_hermite_mods import GaussHermiteBasis
from quality_metrics import QualityMetrics


def save_image(image_array, filename):
    norm = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    norm = (norm * 255).astype(np.uint8)
    Image.fromarray(norm).save(filename)


def run():

    N = int(input("N (степень двойки): "))
    K = int(input("Число коэффициентов: "))

    print("Загрузка изображения...")

    img = data.moon()
    img = resize(img, (N, N), preserve_range=True)
    
    """img = data.camera()
    img = resize(img, (N, N), preserve_range=True)"""

    save_image(img, "original.png")

    # -------------------
    # HADAMARD
    # -------------------
    print("Hadamard...")

    had = HadamardBasis(N)
    coeffs_h = had.get_n_coeffs(img, K, order="russian_doll")
    rec_h = had.recovery(coeffs_h)

    save_image(rec_h, "hadamard.png")

    # -------------------
    # HG
    # -------------------
    print("Gauss-Hermite...")

    gh = GaussHermiteBasis(N, w0=1.0, window_factor=8)
    coeffs_g = gh.get_n_coeffs(img, K, order="hg")
    rec_g = gh.recovery(coeffs_g)

    save_image(rec_g, "gauss_hermite.png")

    # -------------------
    # METRICS
    # -------------------
    print("\n--- METRICS ---")

    print("Hadamard RMSE %:", QualityMetrics.rmse_percent(img, rec_h))
    print("HG RMSE %:", QualityMetrics.rmse_percent(img, rec_g))
    print("Hadamard SSIM %:", QualityMetrics.ssim(img, rec_h))
    print("HG SSIM %:", QualityMetrics.ssim(img, rec_g))

if __name__ == "__main__":
    run()
