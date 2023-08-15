#!/bin/env python
import pytorch_lightning as pl
import torch.profiler
from pytorch_lightning.profilers import PyTorchProfiler

from models.nerf_pl import NeRF_pl
from opt import get_opts

# import nvidia_dlprof_pytorch_nvtx
# nvidia_dlprof_pytorch_nvtx.init()


def main():
    args = get_opts()
    system = NeRF_pl(args)

    logger = pl.loggers.TensorBoardLogger(save_dir=args.logs_dir,
                                          name=args.exp_name,
                                          default_hp_metric=False)

    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.logs_dir),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
        trace_memory=False,
        profile_memory=True,
        emit_nvtx=True,
        # export_to_chrome=True,
        use_cpu=True,
        # use_cuda=True
    )

    ckpt_callback = pl.callbacks.ModelCheckpoint(dirpath="{}/{}".format(args.ckpts_dir, args.exp_name),
                                                 filename="{epoch:d}",
                                                 monitor="val/psnr",
                                                 mode="max",
                                                 save_top_k=-1,
                                                 every_n_epochs=args.save_every_n_epochs)
    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         logger=logger,
                         callbacks=[ckpt_callback],
                         devices=args.gpu_id,
                         # deterministic=True, # RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'.
                         benchmark=True,
                         # weights_summary=None,  # pass a ModelSummary callback with max_depth instead
                         num_sanity_val_steps=2,
                         check_val_every_n_epoch=1,
                         profiler=profiler)

    trainer.fit(system, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
