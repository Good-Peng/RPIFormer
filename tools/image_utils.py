import math
from pathlib import Path

import cv2
import numpy as np
import torch


def calculate_psnr(img1, img2, border=0):
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def PSNR(img1, img2):
    mse_ = np.mean((img1 - img2) ** 2)
    if mse_ == 0:
        return 100
    return 10 * math.log10(1 / mse_)


def calculate_ssim(img1, img2, border=0):
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    if img1.ndim == 3:
        if img1.shape[2] == 3:
            return np.mean([ssim(img1[:, :, i], img2[:, :, i]) for i in range(3)])
        if img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    raise ValueError("Wrong input image dimensions.")


def ssim(img1, img2):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def load_img(filepath):
    data = np.fromfile(str(filepath), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(filepath)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_img(filepath, img):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ext = filepath.suffix if filepath.suffix else ".png"
    ok, buf = cv2.imencode(ext, bgr)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {filepath}")
    buf.tofile(str(filepath))


def visualization(feature, save_path, type="max", colormap=cv2.COLORMAP_JET):
    feature = feature.cpu().numpy()
    if type == "mean":
        feature = np.mean(feature, axis=0)
    else:
        feature = np.max(feature, axis=0)
    normed_feat = (feature - feature.min()) / (feature.max() - feature.min() + 1e-12)
    normed_feat = (normed_feat * 255).astype("uint8")
    color_feat = cv2.applyColorMap(normed_feat, colormap)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(save_path.suffix if save_path.suffix else ".png", color_feat)
    if not ok:
        raise RuntimeError(f"Failed to encode visualization: {save_path}")
    buf.tofile(str(save_path))


def my_summary(test_model, h=256, w=256, c=3, n=1):
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as e:
        raise ImportError("`fvcore` is required for FLOPs analysis.") from e

    model = test_model.cuda()
    print(model)
    inputs = torch.randn((n, c, h, w)).cuda()
    flops = FlopCountAnalysis(model, inputs)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"GMac:{flops.total() / (1024 * 1024 * 1024)}")
    print(f"Params:{n_param}")


def calculate_lpips(img1, img2, border=0):
    try:
        import lpips
    except ImportError:
        print("Warning: LPIPS not available. Please install the `lpips` package.")
        return 0.0

    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")

    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    img1_tensor = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
    img2_tensor = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1

    loss_fn = lpips.LPIPS(net="alex", verbose=False)
    with torch.no_grad():
        lpips_value = loss_fn(img1_tensor, img2_tensor).item()
    return lpips_value
