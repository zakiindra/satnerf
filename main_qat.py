import os

import pytorch_lightning as pl
import torch.profiler
from pytorch_lightning.profilers import PyTorchProfiler
from models.nerf_pl import NeRF_pl
from quantization.model import QuantizedNeRF_pl, convert_to_qat_model
from opt import get_opts


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    args = get_opts()

    # Add QAT specific arguments
    args.enable_qat = getattr(args, "enable_qat", True)
    args.quantization_config = getattr(args, "quantization_config", None)

    if args.enable_qat:
        print("🔧 Initializing Quantization Aware Training...")
        system = QuantizedNeRF_pl(args)
    else:
        print("🔧 Initializing standard NeRF training...")
        system = NeRF_pl(args)

    logger = pl.loggers.TensorBoardLogger(
        save_dir=args.logs_dir, name=args.exp_name, default_hp_metric=False
    )

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{args.ckpts_dir}/{args.exp_name}",
        filename="{epoch:d}",
        monitor="val/psnr",
        mode="max",
        save_top_k=-1,
        every_n_epochs=args.save_every_n_epochs,
    )

    trainer = pl.Trainer(
        max_steps=args.max_train_steps,
        logger=logger,
        callbacks=[ckpt_callback],
        devices=args.gpu_id,
        benchmark=True,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=1,
        profiler="simple",
    )

    trainer.fit(system, ckpt_path=args.ckpt_path)

    # Export quantized model if QAT was used
    if args.enable_qat:
        export_dir = f"{args.logs_dir}/{args.exp_name}/quantized_exports"
        onnx_path = f"{export_dir}/model_quantized.onnx"

        print("📦 Exporting quantized model...")

        # Ensure model is properly prepared for export
        system.cuda()  # Move to GPU
        system.eval()  # Set to eval mode

        try:
            system.export_quantized_model(onnx_path, format="onnx")
        except Exception as e:
            print(f"ONNX export failed: {e}")
            print(
                "This might be due to model complexity. Consider simplifying the export."
            )

        # Export TensorRT if available
        try:
            import torch._dynamo as dynamo

            with dynamo.disable():
                trt_path = f"{export_dir}/model_quantized.trt"
                system.export_quantized_model(trt_path, format="tensorrt")
        except Exception as e:
            print(f"Warning: TensorRT export failed: {e}")


if __name__ == "__main__":
    main()

