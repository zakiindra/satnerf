#!/bin/env python
import pytorch_lightning as pl

from models.nerf_pl import NeRF_pl
from opt import get_opts


def main():
    args = get_opts()
    system = NeRF_pl(args)

    logger = pl.loggers.TensorBoardLogger(save_dir=args.logs_dir,
                                          name=args.exp_name,
                                          default_hp_metric=False)

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
                         profiler="simple")

    trainer.fit(system, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
