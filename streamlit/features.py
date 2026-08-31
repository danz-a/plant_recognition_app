"""Classical feature extraction for the demo - the SAME code as the notebooks.

Two feature vectors are used by the classical models:

  global6  - the six whole-image summaries from Deliverable 1 (LogReg, RF-6):
             file_size_kb, brightness, mean_R, mean_G, mean_B, green_frac
  feat160  - global6 + RGB/HSV colour histograms + GLCM texture (RF-160)

Both are ported line-for-line from Step3_1_PlantVillage_Baseline_Models /
Step3_1_PlantVillage_RandomForest. The only change: the functions take a
PIL image and the file size (in KB) instead of a path, so they also work
for an uploaded file. Nothing here is fitted or trained.
"""
import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import graycomatrix, graycoprops

GLOBAL6_COLS = ["file_size_kb", "brightness", "mean_R", "mean_G", "mean_B", "green_frac"]


def _norm_hist(values, bins, rng):
    h = np.histogram(values, bins=bins, range=rng)[0].astype("float64")
    s = h.sum()
    return h / s if s > 0 else h


def global6(pil_img, file_size_kb: float) -> np.ndarray:
    """Six global features on the FULL-resolution image (Deliverable 1 definition)."""
    a0 = np.asarray(pil_img.convert("RGB")).astype("float64")
    r, g, b = (a0[:, :, i].mean() for i in range(3))
    br = a0[:, :, :3].mean()
    return np.array([file_size_kb, br, r, g, b, g / (r + g + b)], dtype="float64")


def feat160(pil_img, file_size_kb: float, cfg: dict) -> np.ndarray:
    """RGB hist + HSV hist + GLCM texture (on a 64x64 copy) + global6 (full res).

    cfg comes from models/classical_spec.json - feat_size, rgb_bins, hsv_bins,
    glcm_levels, glcm_dist, glcm_props - so the app can never drift from the
    values the forest was trained with.
    """
    feat_size   = int(cfg["feat_size"])
    rgb_bins    = int(cfg["rgb_bins"])
    hsv_bins    = int(cfg["hsv_bins"])
    glcm_levels = int(cfg["glcm_levels"])
    glcm_dist   = list(cfg["glcm_dist"])
    glcm_props  = list(cfg["glcm_props"])
    glcm_angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    a0 = np.asarray(pil_img.convert("RGB")).astype("float64")
    r, g, b = (a0[:, :, i].mean() for i in range(3))
    br = a0[:, :, :3].mean()
    g6 = np.array([file_size_kb, br, r, g, b, g / (r + g + b)])

    arr = np.asarray(Image.fromarray(a0.astype("uint8")).resize((feat_size, feat_size))).astype("float64")
    rgb_hist = np.concatenate([_norm_hist(arr[:, :, c], rgb_bins, (0, 255)) for c in range(3)])
    hsv = rgb2hsv(arr / 255.0)
    hsv_hist = np.concatenate([_norm_hist(hsv[:, :, c], hsv_bins, (0, 1)) for c in range(3)])
    gray = (rgb2gray(arr / 255.0) * (glcm_levels - 1)).astype("uint8")
    glcm = graycomatrix(gray, distances=glcm_dist, angles=glcm_angles,
                        levels=glcm_levels, symmetric=True, normed=True)
    prop = {p: graycoprops(glcm, p).mean(axis=1) for p in glcm_props}   # each -> (n_dist,)
    glcm_feats = np.array([prop[p][di] for di in range(len(glcm_dist)) for p in glcm_props])

    return np.concatenate([rgb_hist, hsv_hist, glcm_feats, g6]).astype("float32")
