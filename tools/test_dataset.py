import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.util import img_as_ubyte
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.utils.options import parse
from tools import image_utils


def self_ensemble(x, model):

    def forward_transformed(inp, hflip, vflip, rotate, net):
        if hflip:
            inp = torch.flip(inp, (-2,))
        if vflip:
            inp = torch.flip(inp, (-1,))
        if rotate:
            inp = torch.rot90(inp, dims=(-2, -1))
        out = net(inp)
        if rotate:
            out = torch.rot90(out, dims=(-2, -1), k=3)
        if vflip:
            out = torch.flip(out, (-1,))
        if hflip:
            out = torch.flip(out, (-2,))
        return out

    preds = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rotate in [False, True]:
                preds.append(forward_transformed(x, hflip, vflip, rotate, model))
    return torch.mean(torch.stack(preds), dim=0)


def unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("params", "state_dict", "model", "net", "netG", "G"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("Unsupported checkpoint format.")


def get_dataset_opt(opt):
    if "test" in opt["datasets"]:
        dataset_opt = opt["datasets"]["test"]
    elif "val" in opt["datasets"]:
        dataset_opt = opt["datasets"]["val"]
    else:
        raise KeyError(
            "The option file must contain `datasets.test` or `datasets.val`."
        )
    dataset_opt["phase"] = "test"
    if dataset_opt.get("scale") is None:
        dataset_opt["scale"] = 1
    return dataset_opt


def main():
    parser = argparse.ArgumentParser(
        description="Test RPIFormer on a paired low-light dataset."
    )
    parser.add_argument(
        "--opt", type=str, required=True, help="Path to an option YAML file."
    )
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to the model checkpoint."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name used for output folder naming.",
    )
    parser.add_argument(
        "--result_dir", type=str, default="results", help="Directory for test results."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Optional flat output directory for restored images.",
    )
    parser.add_argument(
        "--gpus", type=str, default="0", help="CUDA visible devices, e.g. `0` or `0,1`."
    )
    parser.add_argument(
        "--GT_mean",
        action="store_true",
        help="Rectify output brightness using GT mean (not recommended).",
    )
    parser.add_argument(
        "--self_ensemble", action="store_true", help="Apply x8 self-ensemble inference."
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print(f"export CUDA_VISIBLE_DEVICES={args.gpus}")
    use_cuda = torch.cuda.is_available() and args.gpus != ""
    opt = parse(args.opt, is_train=False)
    opt["dist"] = False
    if not use_cuda:
        opt["num_gpu"] = 0
    model = create_model(opt).net_g
    checkpoint = torch.load(args.weights, map_location="cpu")
    state_dict = unwrap_state_dict(checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        prefixed_state_dict = {f"module.{k}": v for (k, v) in state_dict.items()}
        model.load_state_dict(prefixed_state_dict, strict=True)
    if use_cuda:
        model = model.cuda()
        model = nn.DataParallel(model)
    model.eval()
    print(f"===> Testing using weights: {args.weights}")
    dataset_opt = get_dataset_opt(opt)
    dataset = create_dataset(dataset_opt)
    dataloader = create_dataloader(
        dataset,
        dataset_opt,
        num_gpu=opt.get("num_gpu", 1),
        dist=False,
        sampler=None,
        seed=opt.get("manual_seed"),
    )
    factor = 4
    config_name = Path(args.opt).stem
    checkpoint_name = Path(args.weights).stem
    result_dir = Path(args.result_dir) / args.dataset / config_name / checkpoint_name
    result_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    is_sid = dataset_opt["type"] == "Dataset_SIDImage"
    result_dir_input = Path(args.result_dir) / args.dataset / "input"
    result_dir_gt = Path(args.result_dir) / args.dataset / "gt"
    if is_sid:
        result_dir_input.mkdir(parents=True, exist_ok=True)
        result_dir_gt.mkdir(parents=True, exist_ok=True)
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    with torch.inference_mode():
        for data_batch in tqdm(dataloader, total=len(dataset)):
            if use_cuda:
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
            input_tensor = data_batch["lq"]
            target = data_batch["gt"].cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            input_save = data_batch["lq"].cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            inp_path = data_batch["lq_path"][0]
            if use_cuda:
                input_tensor = input_tensor.cuda(non_blocking=True)
            h, w = (input_tensor.shape[2], input_tensor.shape[3])
            H = (h + factor) // factor * factor
            W = (w + factor) // factor * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_tensor = F.pad(input_tensor, (0, padw, 0, padh), "reflect")
            if h < 3000 and w < 3000:
                restored = (
                    self_ensemble(input_tensor, model)
                    if args.self_ensemble
                    else model(input_tensor)
                )
            else:
                input_1 = input_tensor[:, :, :, 1::2]
                input_2 = input_tensor[:, :, :, 0::2]
                if args.self_ensemble:
                    restored_1 = self_ensemble(input_1, model)
                    restored_2 = self_ensemble(input_2, model)
                else:
                    restored_1 = model(input_1)
                    restored_2 = model(input_2)
                restored = torch.zeros_like(input_tensor)
                restored[:, :, :, 1::2] = restored_1
                restored[:, :, :, 0::2] = restored_2
            restored = restored[:, :, :h, :w]
            restored = (
                torch.clamp(restored, 0, 1).cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            )
            if args.GT_mean:
                mean_restored = cv2.cvtColor(
                    restored.astype(np.float32), cv2.COLOR_RGB2GRAY
                ).mean()
                mean_target = cv2.cvtColor(
                    target.astype(np.float32), cv2.COLOR_RGB2GRAY
                ).mean()
                restored = np.clip(
                    restored * (mean_target / max(mean_restored, 1e-08)), 0, 1
                )
            psnr_scores.append(image_utils.PSNR(target, restored))
            ssim_scores.append(
                image_utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored))
            )
            lpips_scores.append(
                image_utils.calculate_lpips(
                    img_as_ubyte(target), img_as_ubyte(restored)
                )
            )
            save_path = (
                output_dir / f"{Path(inp_path).stem}.png" if output_dir else None
            )
            if is_sid:
                type_id = Path(inp_path).parent.name
                save_path = result_dir / type_id / f"{Path(inp_path).stem}.png"
                image_utils.save_img(
                    result_dir_input / type_id / f"{Path(inp_path).stem}.png",
                    img_as_ubyte(input_save),
                )
                image_utils.save_img(
                    result_dir_gt / type_id / f"{Path(inp_path).stem}.png",
                    img_as_ubyte(target),
                )
            elif save_path is None:
                save_path = result_dir / f"{Path(inp_path).stem}.png"
            image_utils.save_img(save_path, img_as_ubyte(restored))
    print(f"PSNR: {np.mean(np.array(psnr_scores)):.6f}")
    print(f"SSIM: {np.mean(np.array(ssim_scores)):.6f}")
    print(f"LPIPS: {np.mean(np.array(lpips_scores)):.6f}")


if __name__ == "__main__":
    main()
