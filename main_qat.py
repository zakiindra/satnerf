import argparse
import json
import os

import pytorch_lightning as pl
import torch.profiler
from pytorch_lightning.profilers import PyTorchProfiler

from eval_satnerf import load_nerf
from quantization.model import QuantizedNeRF_pl
from opt import get_opts


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # args = get_opts()
    # 
    # # Add QAT specific arguments
    # args.enable_qat = getattr(args, "enable_qat", True)
    # args.quantization_config = getattr(args, "quantization_config", None)
    # 
    # if args.enable_qat:
    #     print("🔧 Initializing Quantization Aware Training...")
    #     system = QuantizedNeRF_pl(args)
    # else:
    #     print("🔧 Initializing standard NeRF training...")
    #     system = NeRF_pl(args)
    # 
    # logger = pl.loggers.TensorBoardLogger(
    #     save_dir=args.logs_dir, name=args.exp_name, default_hp_metric=False
    # )
    # 
    # ckpt_callback = pl.callbacks.ModelCheckpoint(
    #     dirpath=f"{args.ckpts_dir}/{args.exp_name}",
    #     filename="{epoch:d}",
    #     monitor="val/psnr",
    #     mode="max",
    #     save_top_k=-1,
    #     every_n_epochs=args.save_every_n_epochs,
    # )
    # 
    # trainer = pl.Trainer(
    #     max_steps=args.max_train_steps,
    #     logger=logger,
    #     callbacks=[ckpt_callback],
    #     devices=args.gpu_id,
    #     benchmark=True,
    #     num_sanity_val_steps=2,
    #     check_val_every_n_epoch=1,
    #     profiler="simple",
    # )
    # 
    # trainer.fit(system, ckpt_path=args.ckpt_path)

    logs_dir = "/data/zis35724/jupyter/satnerf-old/exp-all/JAX_412_ds1_2gpu_batch4096_satnerf/logs"
    run_id = "2023-09-19_02-43-50_JAX_412_ds1_2gpu_batch4096_satnerf"
    ckpt_dir = "/data/zis35724/jupyter/satnerf-old/exp-all/JAX_412_ds1_2gpu_batch4096_satnerf/checkpoints"
    print(os.path.join(logs_dir, run_id))
    with open('{}/opts.json'.format(os.path.join(logs_dir, run_id)), 'r') as f:
        args = argparse.Namespace(**json.load(f))
        print(args)

    workdir = "/home/myid/zis35724/jupyter/satnerf"

    args.enable_qat = getattr(args, "enable_qat", True)
    args.root_dir = "/data/zis35724/jupyter/satnerf/datasets/Track3-preprocess/JAX_412/ba"
    args.img_dir = "/data/zis35724/jupyter/satnerf/datasets/Track3-preprocess/JAX_412/ba/crops"
    args.gt_dir = "/data/zis35724/jupyter/satnerf/datasets/DFC2019/Track3-Truth"
    args.cache_dir = "/data/zis35724/jupyter/satnerf/datasets/Track3-preprocess/JAX_412/ba/cache"
    args.max_epochs = 2
    args.exp_name = "satnerf_ptq_qat"
    args.logs_dir = f"{workdir}/exp-qat/JAX_412_satnerf_ptq_qat/logs"
    args.ckpt_dir = f"{workdir}/exp-qat/JAX_412_satnerf_ptq_qat/checkpoints"
    epoch_number = 23

    models = load_nerf(run_id, logs_dir, ckpt_dir, epoch_number)
    system = QuantizedNeRF_pl(args, models=models)

    # logger = pl.loggers.TensorBoardLogger(
    #     save_dir=args.logs_dir,
    #     name=args.exp_name,
    #     default_hp_metric=False
    # )
    #
    #
    # ckpt_callback = pl.callbacks.ModelCheckpoint(
    #     dirpath=args.ckpt_dir,
    #     filename="{epoch:d}",
    #     monitor="val/psnr",
    #     mode="max",
    #     save_top_k=-1,
    #     every_n_epochs=args.save_every_n_epochs,
    # )
    #
    # trainer = pl.Trainer(
    #     max_epochs=args.max_epochs,
    #     logger=logger,
    #     callbacks=[ckpt_callback],
    #     devices=1,
    #     benchmark=True,
    #     num_sanity_val_steps=2,
    #     check_val_every_n_epoch=1,
    #     profiler="simple",
    # )
    #
    # trainer.fit(system)


    # Export quantized model if QAT was used
    if args.enable_qat:
        export_dir = f"{args.logs_dir}/{args.exp_name}/quantized_exports"
        onnx_path = f"{export_dir}/model_quantized.onnx"

        print("📦 Exporting quantized model...")

        # Ensure model is properly prepared for export
        system.cuda()  # Move to GPU
        system.eval()  # Set to eval mode

        # try:
        os.makedirs(export_dir)
        system.export_onnx(onnx_path)
        # except Exception as e:
        #    print(f"ONNX export failed: {e}")

        # Export TensorRT if available
        # try:
        #    import torch._dynamo as dynamo
        #    with dynamo.disable():
        #        trt_path = f"{export_dir}/model_quantized.trt"
        #        system.export_quantized_model(trt_path, format="tensorrt")
        # except Exception as e:
        #    print(f"Error: TensorRT export failed: {e}")


if __name__ == "__main__":
    main()

