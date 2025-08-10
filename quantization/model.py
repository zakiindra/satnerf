#!/usr/bin/env python3
"""
Quantization Aware Training integration for NeRF using NVIDIA ModelOpt
"""

import os

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Union
import json
import argparse
from collections import defaultdict

# NVIDIA ModelOpt imports
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizeConfig
from modelopt.torch.quantization import utils as quant_utils

from models.satnerf_trt import SatNeRF

MODELOPT_AVAILABLE = True

# TensorRT imports (optional)
import tensorrt as trt
import torch_tensorrt

TENSORRT_AVAILABLE = True

class QuantizedNeRF_pl(pl.LightningModule):
    """NeRF with Quantization Aware Training support"""

    def __init__(self, args, models=None, base_model: Optional[pl.LightningModule] = None):
        super().__init__()
        self.args = args
        self.qat_enabled = getattr(args, 'enable_qat', False)
        self.quantization_config = getattr(args, 'quantization_config', None)
        self.models = models

        # Initialize base model or create new one
        if base_model is not None:
            # Copy attributes from base model
            for attr_name in dir(base_model):
                if not attr_name.startswith('_') and hasattr(base_model, attr_name):
                    attr_value = getattr(base_model, attr_name)
                    if not callable(attr_value):
                        setattr(self, attr_name, attr_value)

            # Copy models
            self.models = base_model.models if hasattr(base_model, 'models') else {}
        else:
            # Initialize from scratch (fallback to original NeRF_pl initialization)
            self._init_base_components()

        # Initialize quantization if enabled
        if self.qat_enabled and MODELOPT_AVAILABLE:
            self._setup_quantization()

    def _init_base_components(self):
        """Initialize base NeRF components (fallback method)"""
        from metrics import load_loss, DepthLoss, SNerfLoss

        self.loss = load_loss(self.args)
        self.depth = self.args.ds_lambda > 0

        if self.depth:
            self.depth_loss = DepthLoss(lambda_ds=self.args.ds_lambda)
            self.ds_drop = np.round(self.args.ds_drop * self.args.max_train_steps)

        self.define_models()
        self.val_im_dir = "{}/{}/val".format(self.args.logs_dir, self.args.exp_name)
        self.train_im_dir = "{}/{}/train".format(self.args.logs_dir, self.args.exp_name)
        self.train_steps = 0

        self.use_ts = False
        if self.args.model == "sat-nerf":
            self.loss_without_beta = SNerfLoss(lambda_sc=self.args.sc_lambda)
            self.use_ts = True

    def define_models(self):
        """Define NeRF models"""
        from models import load_model

        print(self.models)
        if self.models == None:
            self.models = {}
            self.nerf_coarse = SatNeRF(layers=self.args.fc_layers,
                                       feat=self.args.fc_units,
                                       t_embedding_dims=self.args.t_embbeding_tau)
            # self.nerf_coarse = load_model(self.args)
            self.models['coarse'] = self.nerf_coarse
    
            if self.args.n_importance > 0:
                self.nerf_fine = load_model(self.args)
                self.models['fine'] = self.nerf_fine
    
            if self.args.model == "sat-nerf":
                self.embedding_t = torch.nn.Embedding(
                    self.args.t_embbeding_vocab,
                    self.args.t_embbeding_tau
                )
                self.models["t"] = self.embedding_t

    def _setup_quantization(self):
        """Setup quantization for all models"""

        print("setup quantization")
        if not MODELOPT_AVAILABLE:
            raise RuntimeError("ModelOpt is required for QAT but not available")

        # Default quantization config
        default_config = {
            "quant_cfg": {
                "*weight_quantizer": {"num_bits": 8, "axis": None},
                "*input_quantizer": {"num_bits": 8, "axis": None},
                "*output_quantizer": {"num_bits": 8, "axis": None},
            },
            "algorithm": "max",
        }

        # Use custom config if provided
        quant_config = self.quantization_config or default_config

        # Quantize each model
        quantized_models = {}
        for model_name, model in self.models.items():
            if model_name == 't':  # Skip embedding layer for now
                quantized_models[model_name] = model
                continue

            print(f"Setting up quantization for {model_name} model...")

            # Prepare model for quantization
            model.eval()  # Set to eval mode for QAT setup

            # Create quantization config
            # config = QuantizeConfig(quant_config)

            forward_loop_fn = self.get_forward_fn()

            # Apply quantization
            quantized_model = mtq.quantize(model, quant_config, forward_loop=forward_loop_fn)

            # Enable training mode for QAT
            quantized_model.train()

            quantized_models[model_name] = quantized_model

            print(f"✓ Quantization setup complete for {model_name}")

        # Update models dictionary
        self.models = quantized_models

        # Update individual model references
        if "coarse" in self.models:
            self.nerf_coarse = self.models["coarse"]
        if "fine" in self.models:
            self.nerf_fine = self.models["fine"]
        if "t" in self.models:
            self.embedding_t = self.models["t"]

        print(self.models["t"])

    def forward(self, rays, ts):
        """Forward pass with quantized models"""
        from .rendering_trt import render_rays

        chunk_size = self.args.chunk
        batch_size = rays.shape[0]

        results = defaultdict(list)
        metadata = None
        for i in range(0, batch_size, chunk_size):
            rendered_ray_chunks, metadata = render_rays(
                self.models,
                self.args,
                rays[i : i + chunk_size],
                ts[i : i + chunk_size] if ts is not None else None,
            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results, metadata

    def training_step(self, batch, batch_nb):
        """Training step with QAT support"""
        import train_utils
        import metrics

        self.log("lr", train_utils.get_learning_rate(self.optimizer))
        self.train_steps += 1

        rays = batch["color"]["rays"]
        rgbs = batch["color"]["rgbs"]
        ts = None if not self.use_ts else batch["color"]["ts"].squeeze()

        results, _ = self(rays, ts)

        # Loss computation (same as original)
        if 'beta_coarse' in results and self.current_epoch < 2:
            loss, loss_dict = self.loss_without_beta(results, rgbs)
        else:
            loss, loss_dict = self.loss(results, rgbs)
        self.args.noise_std *= 0.9

        # Depth loss if enabled
        if self.depth:
            tmp = self(batch["depth"]["rays"], batch["depth"]["ts"].squeeze())
            kp_depths = torch.flatten(batch["depth"]["depths"][:, 0])
            kp_weights = 1. if self.args.ds_noweights else torch.flatten(batch["depth"]["depths"][:, 1])
            loss_depth, tmp = self.depth_loss(tmp, kp_depths, kp_weights)
            if self.train_steps < self.ds_drop:
                loss += loss_depth
            for k in tmp.keys():
                loss_dict[k] = tmp[k]

        self.log("train/loss", loss)
        typ = "fine" if "rgb_fine" in results else "coarse"

        with torch.no_grad():
            psnr_ = metrics.psnr(results[f"rgb_{typ}"], rgbs)
            self.log("train/psnr", psnr_)

        for k in loss_dict.keys():
            self.log("train/{}".format(k), loss_dict[k])

        self.log("train_psnr", psnr_, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss, "train/psnr": psnr_}

    def validation_step(self, batch, batch_nb):
        """Validation step - same as original but with quantized models"""
        import train_utils
        import metrics
        from eval_satnerf import save_nerf_output_to_images, predefined_val_ts
        from sat_utils import compute_mae_and_save_dsm_diff
        import datetime

        rays, rgbs = batch["rays"], batch["rgbs"]
        rays = rays.squeeze()
        rgbs = rgbs.squeeze()

        if self.args.model == "sat-nerf":
            print(predefined_val_ts)
            print(batch["src_id"][0])
            t = predefined_val_ts(batch["src_id"][0])
            print(t)
            ts = t * torch.ones(rays.shape[0], 1).long().cuda().squeeze()
        else:
            ts = None

        results, metadata = self(rays, ts)
        self.metadata = metadata
        loss, loss_dict = self.loss(results, rgbs)

        self.is_validation_image = True
        if self.args.data == 'sat' and batch_nb == 0:
            self.is_validation_image = False

        typ = "fine" if "rgb_fine" in results else "coarse"
        if "h" in batch and "w" in batch:
            W, H = batch["w"], batch["h"]
        else:
            W = H = int(torch.sqrt(torch.tensor(rays.shape[0]).float()))

        img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu()
        img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()
        depth = train_utils.visualize_depth(results[f'depth_{typ}'].view(H, W))
        stack = torch.stack([img_gt, img, depth])

        split = 'val' if self.is_validation_image else 'train'
        sample_idx = batch_nb - 1 if self.is_validation_image else batch_nb
        self.logger.experiment.add_images(f'{split}_{sample_idx}/GT_pred_depth', stack, self.global_step)

        # Save validation images
        epoch = self.current_epoch
        save = not bool(epoch % self.args.save_every_n_epochs)

        if (batch_nb == 0 or batch_nb == 1) and self.args.data == 'sat' and save:
            out_dir = self.val_im_dir if self.is_validation_image else self.train_im_dir
            save_nerf_output_to_images(self.val_dataset[0], batch, results, out_dir, epoch)

        psnr_ = metrics.psnr(results[f"rgb_{typ}"], rgbs)
        ssim_ = metrics.ssim(results[f"rgb_{typ}"].view(1, 3, H, W), rgbs.view(1, 3, H, W))

        # Compute MAE (same as original)
        aoi_id = batch["src_id"][0][:7]
        gt_roi_path = os.path.join(self.args.gt_dir, aoi_id + "_DSM.txt")
        gt_dsm_path = os.path.join(self.args.gt_dir, aoi_id + "_DSM.tif")

        if os.path.exists(gt_roi_path) and os.path.exists(gt_dsm_path):
            depth = results[f"depth_{typ}"]
            unique_identifier = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_path = os.path.join(self.val_im_dir, f"dsm/tmp_pred_dsm_{unique_identifier}.tif")
            _ = self.val_dataset[0].get_dsm_from_nerf_prediction(rays.cpu(), depth.cpu(), dsm_path=out_path)
            mae_ = compute_mae_and_save_dsm_diff(out_path, batch["src_id"][0], self.args.gt_dir, self.val_im_dir, 0,
                                                 save=False)
            os.remove(out_path)
        else:
            mae_ = 0.0

        self.log("val/loss", loss)
        self.log("val/psnr", psnr_)
        self.log("val/ssim", ssim_)
        self.log("val/mae", mae_)

        for k in loss_dict.keys():
            self.log("val/{}".format(k), loss_dict[k])

        return {"loss": loss, "val/psnr": psnr_, "val/ssim": ssim_, "val/mae": mae_}

    def configure_optimizers(self):
        """Configure optimizers for QAT"""
        import train_utils

        parameters = train_utils.get_parameters(self.models)
        self.optimizer = torch.optim.Adam(parameters, lr=self.args.lr, weight_decay=0)

        max_epochs = self.args.max_epochs
        scheduler = train_utils.get_scheduler(
            optimizer=self.optimizer, lr_scheduler="step", num_epochs=max_epochs
        )

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def setup(self, stage: str) -> None:
        """Setup datasets"""
        if stage == "fit":
            from datasets import load_train_dataset, load_val_dataset

            self.train_dataset = [] + load_train_dataset(self.args)
            self.val_dataset = [] + load_val_dataset(self.args)

    def prepare_data(self) -> None:
        """Prepare data"""
        from datasets.satellite_dataset import init_scaling_params, generate_train_cache

        init_scaling_params(self.args.root_dir, float(self.args.img_downscale))
        generate_train_cache(
            self.args.root_dir, self.args.cache_dir, float(self.args.img_downscale)
        )

    def train_dataloader(self):
        """Train dataloader"""
        from torch.utils.data import DataLoader

        a = DataLoader(
            self.train_dataset[0],
            shuffle=True,
            num_workers=16,
            batch_size=self.args.batch_size,
            pin_memory=True,
        )
        loaders = {"color": a}

        if self.depth:
            b = DataLoader(
                self.train_dataset[1],
                shuffle=True,
                num_workers=16,
                batch_size=self.args.batch_size,
                pin_memory=True,
            )
            loaders["depth"] = b

        return loaders

    def val_dataloader(self):
        """Validation dataloader"""
        from torch.utils.data import DataLoader

        return DataLoader(
            self.val_dataset[0],
            shuffle=False,
            num_workers=16,
            batch_size=1,
            pin_memory=True,
        )

    def prepare_for_export(self):
        print("Prepare for export")
        """Prepare model for ONNX export by ensuring device consistency"""
        device = next(self.parameters()).device
        print("prepare for export device", device)

        # Move all components to the same device
        for name, model in self.models.items():
            model.to(device)

        # Ensure model is in eval mode
        self.eval()

        # Run a test forward pass to check for device issues
        test_rays = torch.randn(10, 11, device=device)
        test_ts = torch.randint(0, 10, (10,), device=device) if self.use_ts else None

        # try:
        with torch.no_grad():
            _ = self(test_rays, test_ts)
            # print("✅ Device consistency check passed")
            # return True
        # except Exception as e:
        #     print(f"❌ Device consistency check failed: {e}")
        #     return False
        print("Prepare check done")

    def export_quantized_model(self, export_path: str, format: str = "onnx"):
        """Export quantized model for deployment"""
        if not self.qat_enabled:
            raise RuntimeWarning("QAT is not enabled, exporting regular model")

        # self.prepare_for_export()

        # Ensure model is on GPU and in eval mode
        self.cuda()
        self.eval()

        # Calibrate if not already done
        # print("Calibration checking")
        # if not self._is_calibrated():
        #     print("⚠️ Model not calibrated, running calibration...")
        #     self.calibrate_quantizers(self.val_dataloader(), num_batches=5)

        # # Force calibration regardless of current state
        # print("🔧 Running calibration before export...")
        # import modelopt.torch.quantization as mtq

        # self.calibrate_quantizers(self.val_dataloader(), num_batches=10)

        print("📦 Exporting with ModelOpt...")
        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        # Create dummy input for export
        device = next(self.parameters()).device
        dummy_rays = torch.randn(1024, 11).to(device)
        dummy_ts = torch.randint(0, 10, (1024,)).to(device) if self.use_ts else None

        # # Use ModelOpt's export
        # if format.lower() == "onnx":
        #     torch.onnx.export()
        #     print(f"✓ Model exported to ONNX: {export_path}")

    def _is_calibrated(self):
        """Check if quantizers are calibrated"""
        print("Check calibrated")
        for name, module in self.models['coarse'].named_modules():
            print(name, module)
            if hasattr(module, 'input_quantizer') and hasattr(module.input_quantizer, '_amax'):
                if not hasattr(module.input_quantizer, '_amax') or module.input_quantizer._amax is None:
                    return False
        return True

    def export_onnx(self, export_path: str):
        """Export to ONNX format"""
        # Ensure model is on GPU
        device = next(self.parameters()).device
        print("export_onnx device", device)

        input_xyz = torch.randn(self.metadata["shape"]["input_xyz"], dtype=torch.float32, device=device)
        # input_direction = None  # Originally there is no input_direction parameter
        input_direction = torch.zeros(self.metadata["shape"]["input_xyz"], dtype=torch.float32,
                                      device=device)  # Dummy zeros tensor for ONNX conversion
        input_sun_direction = torch.randn(self.metadata["shape"]["input_sun_dir"], dtype=torch.float32, device=device)
        input_t = torch.randn(self.metadata["shape"]["input_t"], dtype=torch.float32, device=device)

        print(f"ONNX Conversion with the below shapes:")
        print(f"xyz_:\t\t\t{self.metadata['shape']['input_xyz']}")
        print(f"direction (zeroes):\t{self.metadata['shape']['input_xyz']}")
        print(f"sun direction:\t\t{self.metadata['shape']['input_sun_dir']}")
        print(f"input t:\t\t{self.metadata['shape']['input_t']}")

        self.eval()

        # Prepare inputs
        inputs = (input_xyz, input_direction, input_sun_direction, input_t)
        input_names = ["input_xyz", "input_dir", "input_sun_dir", "input_t"]
        output_names = ["output"]
        dynamic_axes = {
            "input_xyz": {0: "num_points"},
            "input_dir": {0: "num_points"},
            "input_sun_dir": {0: "num_points"},
            "input_t": {0: "num_points"},
            "output": {0: "num_points"}
        }
        # if self.use_ts:
        #     dynamic_axes['transient'] = {0: 'batch_size'}

        # for name, module in self.models['coarse'].named_modules():
        #     print(name, module)

        mtq.calibrate(self.models["coarse"], forward_loop=self.get_forward_fn())

        with torch.no_grad():
            torch.onnx.export(
                self.models['coarse'],
                inputs,
                export_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=13,
                do_constant_folding=True,
                # verbose=False
            )
        print(f"✓ Model exported to ONNX: {export_path}")

    def _export_tensorrt(self, export_path: str):
        """Export to TensorRT format"""
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available")

        # Create dummy input
        dummy_rays = torch.randn(1024, 11).cuda()
        dummy_ts = torch.randint(0, 10, (1024,)).cuda() if self.use_ts else None

        self.eval()

        # Convert to TensorRT
        inputs = [dummy_rays]
        if dummy_ts is not None:
            inputs.append(dummy_ts)

        trt_model = torch_tensorrt.compile(
            self,
            inputs=inputs,
            enabled_precisions={torch.float, torch.half, torch.int8},
            workspace_size=1 << 22,  # 4MB
        )

        # Save TensorRT model
        torch.jit.save(trt_model, export_path)
        print(f"✓ Model exported to TensorRT: {export_path}")

    # def calibrate_quantizers(self, calibration_dataloader, num_batches=10):
    #     """Calibrate quantizers using sample data"""
    #     if not self.qat_enabled:
    #         return
    #
    #     print("🔧 Calibrating quantizers...")
    #     self.eval()
    #
    #     # Enable calibration mode
    #     import modelopt.torch.quantization as mtq
    #
    #     with torch.no_grad():
    #         batch_count = 0
    #         for batch_idx, batch in enumerate(calibration_dataloader):
    #             if batch_count >= num_batches:
    #                 break
    #
    #             # Move to GPU if available
    #             rays = batch["rays"].cuda() if torch.cuda.is_available() else batch["rays"]
    #
    #             # Handle timestamp for sat-nerf
    #             if self.use_ts:
    #                 from eval_satnerf import predefined_val_ts
    #                 t = predefined_val_ts(batch["src_id"][0])
    #                 ts = t * torch.ones(rays.shape[0], 1).long()
    #                 if torch.cuda.is_available():
    #                     ts = ts.cuda()
    #             else:
    #                 ts = None
    #
    #             # Forward pass for calibration
    #             _ = self(rays, ts)
    #             batch_count += 1
    #
    #             print(f"Calibrated batch {batch_count}/{num_batches}")
    #
    #     print("✅ Quantizer calibration complete!")

    def calibrate_quantizers(self, calibration_dataloader, num_batches=10):
        """Calibrate quantizers using sample data"""
        if not self.qat_enabled:
            return

        print("🔧 Calibrating quantizers...")

        # Set model to calibration mode
        import modelopt.torch.quantization as mtq

        self.eval()

        with torch.no_grad():
            batch_count = 0
            for batch_idx, batch in enumerate(calibration_dataloader):
                if batch_count >= num_batches:
                    break

                try:
                    # Move to GPU if available
                    rays = batch["rays"]
                    if torch.cuda.is_available():
                        rays = rays.cuda()
                    rays = rays.squeeze()

                    # Handle timestamp for sat-nerf
                    if self.use_ts:
                        from eval_satnerf import predefined_val_ts

                        t = predefined_val_ts(batch["src_id"][0])
                        ts = t * torch.ones(rays.shape[0], 1).long()
                        if torch.cuda.is_available():
                            ts = ts.cuda().squeeze()
                    else:
                        ts = None

                    # Forward pass for calibration - use smaller chunks
                    chunk_size = min(1024, rays.shape[0])
                    for i in range(0, rays.shape[0], chunk_size):
                        ray_chunk = rays[i : i + chunk_size]
                        ts_chunk = ts[i : i + chunk_size] if ts is not None else None
                        _ = self(ray_chunk, ts_chunk)

                    batch_count += 1
                    print(f"Calibrated batch {batch_count}/{num_batches}")

                except Exception as e:
                    print(f"Calibration error on batch {batch_count}: {e}")
                    continue

        # Disable calibration mode
        print("✅ Quantizer calibration complete!")

    def on_fit_end(self) -> None:
        print("on fit end")

        # Ensure all models are on the same device
        device = next(self.parameters()).device
        for name, model in self.models.items():
            model.to(device)

        for name, param in self.models["t"].named_parameters():
            print(f"Parameter '{name}' device: {param.device}")
        for name, param in self.embedding_t.named_parameters():
            print(f"Parameter '{name}' device: {param.device}")

    def get_forward_fn(self):
        def mtq_forward_fn(model):

            from torch.utils.data import DataLoader
            from datasets import load_train_dataset, load_val_dataset

            print("Do forward fn")
            val_dataset = [] + load_val_dataset(self.args)

            dataloader = DataLoader(
                val_dataset[0],
                shuffle=False,
                num_workers=16,
                batch_size=1,
                pin_memory=True,
            )

            batch_count = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    try:
                        # Move to GPU if available
                        rays = batch["rays"]
                        if torch.cuda.is_available():
                            rays = rays.cuda()
                        rays = rays.squeeze()

                        if self.use_ts:
                            from eval_satnerf import predefined_val_ts

                            t = predefined_val_ts(batch["src_id"][0])
                            ts = t * torch.ones(rays.shape[0], 1).long()
                            if torch.cuda.is_available():
                                ts = ts.cuda()
                            ts = ts.squeeze()
                        else:
                            ts = None

                        # Forward pass for calibration - use smaller chunks
                        chunk_size = min(1024, rays.shape[0])
                        for i in range(0, rays.shape[0], chunk_size):
                            ray_chunk = rays[i : i + chunk_size]
                            ts_chunk = (
                                ts[i : i + chunk_size] if ts is not None else None
                            )
                            _ = self(ray_chunk, ts_chunk)

                        print(f"Calibrated batch {batch_count}")

                    except Exception as e:
                        print(f"Calibration error on batch {batch_count}: {e}")
                        continue

        return mtq_forward_fn


def convert_to_qat_model(original_model: pl.LightningModule, args) -> QuantizedNeRF_pl:
    """Convert existing NeRF model to QAT-enabled version"""
    qat_model = QuantizedNeRF_pl(args, base_model=original_model)
    return qat_model


# Usage example and configuration
def get_qat_config_example():
    """Example QAT configuration"""
    return {
        "quant_cfg": {
            # Weight quantization
            "*weight_quantizer": {
                "num_bits": 8,
                "axis": None,
                "unsigned": False,
            },
            # Input quantization
            "*input_quantizer": {
                "num_bits": 8,
                "axis": None,
                "unsigned": False,
            },
            # Output quantization
            "*output_quantizer": {
                "num_bits": 8,
                "axis": None,
                "unsigned": False,
            },
            # Specific layer configurations
            "fc_net.*": {"enable": True},
            "sigma_from_xyz.*": {"enable": True},
            "rgb_from_xyzdir.*": {"enable": True},
        },
        "algorithm": "max",  # or "entropy", "percentile"
    }


if __name__ == "__main__":
    # Example usage
    print("🔧 NVIDIA NeRF Quantization Aware Training Integration")
    print("=" * 60)

    if not MODELOPT_AVAILABLE:
        print("❌ NVIDIA ModelOpt not available")
        print("Install with: pip install nvidia-modelopt")
    else:
        print("✅ NVIDIA ModelOpt available")

    if not TENSORRT_AVAILABLE:
        print("⚠️  TensorRT not available (optional)")
        print("Install with: pip install torch-tensorrt")
    else:
        print("✅ TensorRT available")

    print("\n📋 Example QAT Configuration:")
    import json

    print(json.dumps(get_qat_config_example(), indent=2))

    print("\n📖 Usage Instructions:")
    print("1. Add --enable_qat flag to your training arguments")
    print("2. Optionally provide custom quantization config")
    print("3. Train normally - QAT will run automatically")
    print("4. Export quantized models for deployment")
