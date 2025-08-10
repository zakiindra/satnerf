import time

import torch
import yaml
import os
import json
import train_utils
from models import load_model
from datasets import SatelliteDataset
from rendering_trt import render_rays_trt
from collections import defaultdict
import metrics
import numpy as np
import sat_utils
import train_utils
import argparse
import glob
import shutil

import warnings

warnings.filterwarnings("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner


def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    # print(checkpoint.keys())
    # print(checkpoint['state_dict'].keys())
    # print(model_name, prefixes_to_ignore)
    checkpoint_ = {}
    if 'state_dict' in checkpoint:  # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    # print(checkpoint.keys())
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        # print(k)
        k = k[len(model_name) + 1:]
        # print(k)
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                print('ignore', k)
                break
        else:
            checkpoint_[k] = v
    return checkpoint_


def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)


@torch.no_grad()
# def batched_inference(models, rays, ts, args):
def batched_inference(models, runner, rays, ts, args):
    """Do batched inference on rays using chunk."""
    chunk_size = args.chunk
    batch_size = rays.shape[0]
    # print("[eval_satnerf.batched_inference:53] chunk_size, batch_size, rays.shape, ts", chunk_size, batch_size, rays.shape, ts)

    results = defaultdict(list)
    # for range(0, 489999, 81920)
    for i in range(0, batch_size, chunk_size):
        # print(f"[eval_satnerf.batched_inference:58] render_rays(models, args, ray[{i}:{i + chunk_size}], None)")
        # print(i)
        # print(i+chunk_size)
        rendered_ray_chunks = \
            render_rays_trt(models,
                        runner,
                        args,
                        rays[i:i + chunk_size],
                        ts[i:i + chunk_size] if ts is not None else None)
        # print("[eval_satnerf.batched_inference:64] rendered_ray_chunks.keys(): ", rendered_ray_chunks.keys())

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]
        # break

    for k, v in results.items():
        if results[k][0] is None:
            results[k] = None
        else:
            results[k] = torch.cat(v, 0)

    return results


def load_nerf(run_id, logs_dir, ckpts_dir, epoch_number):
    log_path = os.path.join(run_id, logs_dir)
    with open('{}/opts.json'.format(log_path), 'r') as f:
        args = argparse.Namespace(**json.load(f))

    # checkpoint_path = os.path.join(ckpts_dir, "{}/epoch={}.ckpt".format(run_id, epoch_number))
    # checkpoint_path =
    checkpoint_path = os.path.join("{}/epoch={}.ckpt".format(ckpts_dir, epoch_number))
    print(checkpoint_path)
    print("Using", checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Could not find checkpoint {}".format(checkpoint_path))

    # load models
    models = {}
    # nerf_coarse = load_model(args)
    # load_ckpt(nerf_coarse, checkpoint_path, model_name='nerf_coarse')

    # models["coarse"] = nerf_coarse.cuda().eval()

    # if args.n_importance > 0:
    #     nerf_fine = load_model(args)
    #     load_ckpt(nerf_coarse, checkpoint_path, model_name='nerf_fine')
    #     models['fine'] = nerf_fine.cuda().eval()
    if args.model == "sat-nerf":
        embedding_t = torch.nn.Embedding(args.t_embbeding_vocab, args.t_embbeding_tau)
        load_ckpt(embedding_t, checkpoint_path, model_name='embedding_t')
        models["t"] = embedding_t.cuda().eval()

    models["trt"] = EngineFromBytes(BytesFromPath("/data/rdr78068/satnerf-base/model_int8.engine"))

    return models


def save_nerf_output_to_images(dataset, sample, results, out_dir, epoch_number):
    rays = sample["rays"].squeeze()
    rgbs = sample["rgbs"].squeeze()
    src_id = sample["src_id"][0]
    src_path = os.path.join(dataset.img_dir, src_id + ".tif")

    typ = "fine" if "rgb_fine" in results else "coarse"
    if "h" in sample and "w" in sample:
        W, H = sample["w"][0], sample["h"][0]
    else:
        W = H = int(torch.sqrt(torch.tensor(rays.shape[0]).float()))  # assume squared images
    img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    depth = results[f"depth_{typ}"]

    # save depth prediction
    _, _, alts = dataset.get_latlonalt_from_nerf_prediction(rays.cpu(), depth.cpu())
    out_path = "{}/depth/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
    train_utils.save_output_image(alts.reshape(1, H, W), out_path, src_path)
    # save dsm
    out_path = "{}/dsm/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
    # out_path="/data/rdr78068/dsm/dsm_int8.tif"
    dsm = dataset.get_dsm_from_nerf_prediction(rays.cpu(), depth.cpu(), dsm_path=out_path)
    # save rgb image
    out_path = "{}/rgb/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
    # out_path="/data/rdr78068/rgb/rgb_int8.tif"
    train_utils.save_output_image(img, out_path, src_path)
    # save gt rgb image
    out_path = "{}/gt_rgb/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
    # out_path = ".tif".format(out_dir, src_id, epoch_number)
    # out_path="/data/rdr78068/gtd/gtd_int8.tif"
    # save shadow modelling images
    if f"sun_{typ}" in results:
        s_v = torch.sum(results[f"weights_{typ}"].unsqueeze(-1) * results[f'sun_{typ}'], -2)
        out_path = "{}/sun/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
        train_utils.save_output_image(s_v.view(1, H, W).cpu(), out_path, src_path)
        rgb_albedo = torch.sum(results[f"weights_{typ}"].unsqueeze(-1) * results[f'albedo_{typ}'], -2)
        out_path = "{}/albedo/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
        train_utils.save_output_image(rgb_albedo.cpu().view(H, W, 3).permute(2, 0, 1).cpu(), out_path, src_path)
        if f"ambient_a_{typ}" in results:
            a_rgb = torch.sum(results[f"weights_{typ}"].unsqueeze(-1) * results[f'ambient_a_{typ}'], -2)
            out_path = "{}/ambient_a/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
            train_utils.save_output_image(a_rgb.view(H, W, 3).permute(2, 0, 1).cpu(), out_path, src_path)
            b_rgb = torch.sum(results[f"weights_{typ}"].unsqueeze(-1) * results[f'ambient_b_{typ}'], -2)
            out_path = "{}/ambient_b/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
            train_utils.save_output_image(b_rgb.view(H, W, 3).permute(2, 0, 1).cpu(), out_path, src_path)
        if f"beta_{typ}" in results:
            beta = torch.sum(results[f"weights_{typ}"].unsqueeze(-1) * results[f'beta_{typ}'], -2)
            out_path = "{}/beta/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
            train_utils.save_output_image(beta.view(1, H, W).cpu(), out_path, src_path)
        if f"sky_{typ}" in results:
            sky_rgb = torch.sum(results[f"weights_{typ}"].unsqueeze(-1) * results[f'sky_{typ}'], -2)
            out_path = "{}/sky/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
            train_utils.save_output_image(sky_rgb.cpu().view(H, W, 3).permute(2, 0, 1).cpu(), out_path, src_path)


def find_best_embbeding_for_val_image(models, rays, conf, gt_rgbs, train_indices=None):
    best_ts = None
    best_psnr = 0.

    if train_indices is None:
        train_indices = torch.arange(conf.N_vocab)
    for t in train_indices:
        ts = t.long() * torch.ones(rays.shape[0], 1).long().cuda().squeeze()
        results = batched_inference(models, rays, ts, conf)
        typ = "fine" if "rgb_fine" in results else "coarse"
        psnr_ = metrics.psnr(results[f"rgb_{typ}"].cpu(), gt_rgbs.cpu())
        if psnr_ > best_psnr:
            best_ts = ts
            best_psnr = psnr_

    return best_ts


def find_best_embeddings_for_val_dataset(val_dataset, models, conf, train_indices):
    print("finding best embedding indices for validation dataset...")
    list_of_image_indices = [0]
    for i in np.arange(1, len(val_dataset)):
        sample = val_dataset[i]
        rays, rgbs = sample["rays"].cuda(), sample["rgbs"]
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        src_id = sample["src_id"]
        aoi_id = src_id[:7]
        if aoi_id in ["JAX_068", "JAX_004", "JAX_214"]:
            t = predefined_val_ts(src_id)
        else:
            ts = find_best_embbeding_for_val_image(models, rays, conf, rgbs, train_indices=train_indices)
            t = torch.unique(ts).cpu().numpy()
        print("{}: {}".format(src_id, t))
        list_of_image_indices.append(t)
    print("... done!")
    return list_of_image_indices


def predefined_val_ts(img_id):
    aoi_id = img_id[:7]

    if aoi_id == "JAX_068":
        d = {"JAX_068_013_RGB": 0,
             "JAX_068_002_RGB": 8,
             "JAX_068_012_RGB": 1}  # 3
    elif aoi_id == "JAX_004":
        d = {"JAX_004_022_RGB": 0,
             "JAX_004_014_RGB": 0,
             "JAX_004_009_RGB": 5}
    elif aoi_id == "JAX_214":
        d = {"JAX_214_020_RGB": 0,
             "JAX_214_006_RGB": 8,
             "JAX_214_001_RGB": 18,
             "JAX_214_008_RGB": 2}
    elif aoi_id == "JAX_260":
        d = {"JAX_260_015_RGB": 0,
             "JAX_260_006_RGB": 3,
             "JAX_260_004_RGB": 10}
    # elif aoi_id == "JAX_260":
    #     d = {"JAX_260_008_RGB": 0,
    #          "JAX_260_016_RGB": 3,
    #          "JAX_260_011_RGB": 10}
    elif aoi_id == "JAX_412":
        d = {
            "JAX_412_016_RGB": 0,
            "JAX_412_022_RGB": 8,
            "JAX_412_002_RGB": 18,
            "JAX_412_011_RGB": 2
        }
    elif aoi_id == "JAX_033":
        d = {
            "JAX_033_011_RGB": 0,
            "JAX_033_006_RGB": 3,
            "JAX_033_004_RGB": 10
        }
    elif aoi_id == "JAX_070":
        d = {
            "JAX_070_019_RGB": 0,
            "JAX_070_018_RGB": 3,
            "JAX_070_020_RGB": 10
        }
    elif aoi_id == "JAX_072":
        d = {
            "JAX_072_005_RGB": 0,
            "JAX_072_009_RGB": 3,
            "JAX_072_019_RGB": 10
        }
    elif aoi_id == "JAX_474":
        d = {
            "JAX_474_004_RGB": 0,
            "JAX_474_006_RGB": 8,
            "JAX_474_025_RGB": 18,
            "JAX_474_015_RGB": 2
        }
    elif aoi_id == "JAX_427":
        d = {
            "JAX_427_022_RGB": 0,
            "JAX_427_006_RGB": 8,
            "JAX_427_012_RGB": 18,
            "JAX_427_026_RGB": 2
        }
    elif aoi_id == "JAX_467":
        d = {
            "JAX_467_013_RGB": 0,
            "JAX_467_026_RGB": 8,
            "JAX_467_015_RGB": 18,
            "JAX_467_019_RGB": 2
        }
    elif aoi_id == "JAX_416":
        d = {
            "JAX_416_021_RGB": 0,
            "JAX_416_026_RGB": 8,
            "JAX_416_002_RGB": 18,
            "JAX_416_010_RGB": 2
        }
    elif aoi_id == "JAX_280":
        d = {
            "JAX_280_020_RGB": 0,
            "JAX_280_013_RGB": 8,
            "JAX_280_002_RGB": 18,
            "JAX_280_021_RGB": 2
        }
    elif aoi_id == "JAX_022":
        d = {
            "JAX_022_006_RGB": 0,
            "JAX_022_001_RGB": 3,
            "JAX_022_002_RGB": 10
        }
    elif aoi_id == "JAX_028":
        d = {
            "JAX_028_015_RGB": 0,
            "JAX_028_004_RGB": 3,
            "JAX_028_011_RGB": 10
        }
    elif aoi_id == "OMA_042":
        d = {
            "OMA_042_043_RGB": 3,
            "OMA_042_041_RGB": 3,
            "OMA_042_014_RGB": 3,
            "OMA_042_024_RGB": 3,
            "OMA_042_029_RGB": 3,
            "OMA_042_028_RGB": 3,
        }
    elif aoi_id == "OMA_559":
        d = {
            "OMA_559_022_RGB": 3,
            "OMA_559_023_RGB": 3,
            "OMA_559_020_RGB": 3,
            "OMA_559_005_RGB": 3,
        }
    else:
        return 3
    return d[img_id]


# python3 eval_satnerf.py
# Sat-NeRF -> run_id
# $pretrained_models/JAX_068 -> logs_dir
# out_eval_path/JAX_068 -> output_dir
# 28 -> epoch
# val -> split
# $pretrained_models/JAX_068 -> checkpoints_dir
# $dataset_dir/root_dir/crops_rpcs_ba_v2/JAX_068 -> root_dir
# $dataset_dir/DFC2019/Track3-RGB-crops/JAX_068 -> img_dir
# $dataset_dir/DFC2019/Track3-Truth -> gt_dir

def eval_aoi(run_id, logs_dir, output_dir, epoch_number, split, checkpoints_dir=None, root_dir=None, img_dir=None,
             gt_dir=None):
    with open('{}/opts.json'.format(os.path.join(run_id, logs_dir)), 'r') as f:
        args = argparse.Namespace(**json.load(f))
        print(args)

    if gt_dir is not None:
        assert os.path.isdir(gt_dir)
        args.gt_dir = gt_dir
    if img_dir is not None:
        assert os.path.isdir(img_dir)
        args.img_dir = img_dir
    if root_dir is not None:
        assert os.path.isdir(root_dir)
        args.root_dir = root_dir
    if not os.path.isdir(args.cache_dir):
        args.cache_dir = None

    # load pretrained nerf
    if checkpoints_dir is None:
        checkpoints_dir = args.ckpts_dir
        print(f"Checkpoints dir: {checkpoints_dir}")

    models = load_nerf(run_id, logs_dir, checkpoints_dir, epoch_number - 1)

    # prepare dataset
    dataset = SatelliteDataset(args.root_dir,
                               args.img_dir,
                               split="val",
                               img_downscale=args.img_downscale,
                               cache_dir=args.cache_dir
                               )

    if split == "train":
        with open(os.path.join(args.root_dir, "train.txt"), "r") as f:
            json_files = f.read().split("\n")
        dataset.json_files = [os.path.join(args.root_dir, json_p) for json_p in json_files]
        dataset.all_ids = [i for i, p in enumerate(dataset.json_files)]
        samples_to_eval = np.arange(0, len(dataset))
    else:
        samples_to_eval = np.arange(1, len(dataset))

    psnr, ssim, mae = [], [], []

    with TrtRunner(models["trt"]) as runner:
        print("Warmup stage")
        for i in samples_to_eval:
            sample = dataset[i]
            rays, rgbs = sample["rays"].cuda(), sample["rgbs"]
            rays = rays.squeeze()  # (H*W, 3)
            rgbs = rgbs.squeeze()  # (H*W, 3)
            src_id = sample["src_id"]
            if "h" in sample and "w" in sample:
                W, H = sample["w"], sample["h"]
            else:
                W = H = int(torch.sqrt(torch.tensor(rays.shape[0]).float()))

            ts = None
            if args.model == "sat-nerf":
                if split == "val":
                    t = predefined_val_ts(src_id)
                    ts = t * torch.ones(rays.shape[0], 1).long().cuda().squeeze()
                else:
                    ts = sample["ts"].cuda().squeeze()

            start = time.perf_counter()
            _ = batched_inference(models, runner, rays, ts, args)
            end = time.perf_counter()
            print("Warmup time: ", end - start)

        print("Inference...")
        for i in samples_to_eval:
            sample = dataset[i]
            rays, rgbs = sample["rays"].cuda(), sample["rgbs"]
            rays = rays.squeeze()  # (H*W, 3)
            rgbs = rgbs.squeeze()  # (H*W, 3)
            src_id = sample["src_id"]
            if "h" in sample and "w" in sample:
                W, H = sample["w"], sample["h"]
            else:
                W = H = int(torch.sqrt(torch.tensor(rays.shape[0]).float()))

            ts = None
            if args.model == "sat-nerf":
                if split == "val":
                    t = predefined_val_ts(src_id)
                    ts = t * torch.ones(rays.shape[0], 1).long().cuda().squeeze()
                else:
                    ts = sample["ts"].cuda().squeeze()

            start = time.perf_counter()
            results = batched_inference(models, runner, rays, ts, args)
            end = time.perf_counter()

            print("Inference time: ", end - start)

            for k in sample.keys():
                if torch.is_tensor(sample[k]):
                    sample[k] = sample[k].unsqueeze(0)
                else:
                    sample[k] = [sample[k]]
            out_dir = os.path.join(output_dir, run_id, split)
            os.makedirs(out_dir, exist_ok=True)
            save_nerf_output_to_images(dataset, sample, results, out_dir, epoch_number)

            # image metrics
            typ = "fine" if "rgb_fine" in results else "coarse"
            psnr_ = metrics.psnr(results[f"rgb_{typ}"].cpu(), rgbs.cpu())
            psnr.append(psnr_)
            ssim_ = metrics.ssim(results[f"rgb_{typ}"].view(1, 3, H, W).cpu(), rgbs.view(1, 3, H, W).cpu())
            ssim.append(ssim_)

            # geometry metrics
            pred_dsm_path = "{}/dsm/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
            mae_ = sat_utils.compute_mae_and_save_dsm_diff(pred_dsm_path, src_id, args.gt_dir, out_dir, epoch_number)
            mae.append(mae_)
            print("{}: pnsr {:.3f} / ssim {:.3f} / mae {:.3f}".format(src_id, psnr_, ssim_, mae_))

            # clean files
            in_tmp_path = glob.glob(os.path.join(out_dir, "*rdsm_epoch*.tif"))[0]
            out_tmp_path = in_tmp_path.replace(out_dir, os.path.join(out_dir, "rdsm"))
            os.makedirs(os.path.dirname(out_tmp_path), exist_ok=True)
            shutil.copyfile(in_tmp_path, out_tmp_path)
            os.remove(in_tmp_path)
            in_tmp_path = glob.glob(os.path.join(out_dir, "*rdsm_diff_epoch*.tif"))[0]
            out_tmp_path = in_tmp_path.replace(out_dir, os.path.join(out_dir, "rdsm_diff"))
            os.makedirs(os.path.dirname(out_tmp_path), exist_ok=True)
            shutil.copyfile(in_tmp_path, out_tmp_path)
            os.remove(in_tmp_path)

    # Example chunk size (from args.chunk)
    # device = "cuda:0"
    # input_xyz = torch.randn((1310720, 3), dtype=torch.float32, device=device)
    # # input_direction = None  # Originally there is no input_direction parameter
    # input_direction = torch.zeros((1310720, 3), dtype=torch.float32, device=device)  # Dummy zeros tensor for ONNX conversion
    # input_sun_direction = torch.randn((1310720, 3), dtype=torch.float32, device=device)
    # input_t = torch.randn((1310720, 4), dtype=torch.float32, device=device)

    # Prepare inputs as a tuple (ONNX requires positional arguments)
    # example_inputs = (input_xyz, input_direction, input_sun_direction, input_t)

    # start = time.time()
    # torch.onnx.export(
    #     models["coarse"],
    #     example_inputs,
    #     "model.onnx",  # Output ONNX file
    #     input_names=["input_xyz", "input_dir", "input_sun_dir", "input_t"],
    #     output_names=["output"],
    #     dynamic_axes={
    #         "input_xyz": {0: "num_points"},
    #         "input_dir": {0: "num_points"},
    #         "input_sun_dir": {0: "num_points"},
    #         "input_t": {0: "num_points"},
    #         "output": {0: "num_points"}
    #     },
    # )

    # delta = time.time() - start
    # print(f"ONNX conversion: {delta}")

    print("\nMean PSNR: {:.3f}".format(np.mean(np.array(psnr))))
    print("Mean SSIM: {:.3f}".format(np.mean(np.array(ssim))))
    print("Mean MAE: {:.3f}\n".format(np.mean(np.array(mae))))
    return np.mean(np.array(psnr)), np.mean(np.array(ssim)), np.mean(np.array(mae))


# if __name__ == '__main__':
#     import fire
#     fire.Fire(eval_aoi)
#
#
# for path in glob.glob("exp/*"):
#     if path in ["exp/JAX_072_ds1_2gpu_batch4096_satnerf"]:
#         continue
#     logs_dir = path + "/logs"
#     run_id = os.listdir(logs_dir)[0]
#
#     epochs = [int(e.replace("epoch=", "").replace(".ckpt", ""))
#               for e in os.listdir(path + "/checkpoints/" + run_id)
#               if "tmp_end" not in e]
#
#     print(sorted(epochs))
#     eval_metrics = []
#     for e in sorted(epochs):
#
#         psnr, ssim, mae = eval_aoi(run_id, logs_dir, "eval-mae-checkpoint", int(e)+1, "val")
#         eval_metrics.append(f"{e},{psnr},{ssim},{mae}")
#
#     with open(f"eval-mae-checkpoint/{run_id}/metrics.txt", "w") as f:
#         f.write("\n".join(eval_metrics))
#

if __name__ == "__main__":
    run_id = "/data/exp-all/JAX_260_ds1_2gpu_batch4096_satnerf/"
    logs_dir = "logs/2023-09-25_15-52-11_JAX_260_ds1_2gpu_batch4096_satnerf/"
    epoch_number = 32
    split = "val"
    output_dir = "./exps-eval"

    eval_aoi(run_id, logs_dir, output_dir, epoch_number, split)