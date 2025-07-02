# #!/usr/bin/env python3
# """
# Complete NeRF Quantization Aware Training Utilities and Deployment Tools
# Includes calibration, profiling, deployment, and TensorRT optimization
# """
#
# import os
# import time
# import torch
# import numpy as np
# from typing import Dict, List, Optional, Union, Tuple
# import json
# import argparse
# from pathlib import Path
# import logging
# from dataclasses import dataclass
# import subprocess
# import tempfile
#
# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# try:
#     import modelopt.torch.quantization as mtq
#     from modelopt.torch.quantization.config import QuantizeConfig
#     import modelopt.torch.utils as model_utils
#
#     MODELOPT_AVAILABLE = True
# except ImportError:
#     logger.warning("NVIDIA ModelOpt not available. Install with: pip install nvidia-modelopt")
#     MODELOPT_AVAILABLE = False
#
# try:
#     import polygraphy
#     from polygraphy.backend.trt import CreateConfig, Profile
#     from polygraphy.backend.onnx import OnnxFromPath
#     from polygraphy.backend.trt import EngineFromNetwork, TrtRunner
#     from polygraphy import mod
#
#     POLYGRAPHY_AVAILABLE = True
# except ImportError:
#     logger.warning("Polygraphy not available. Install with: pip install polygraphy")
#     POLYGRAPHY_AVAILABLE = False
#
# try:
#     import tensorrt as trt
#     import torch_tensorrt
#
#     TENSORRT_AVAILABLE = True
# except ImportError:
#     logger.warning("TensorRT not available for deployment optimizations")
#     TENSORRT_AVAILABLE = False
#
#
# @dataclass
# class DeploymentConfig:
#     """Configuration for model deployment"""
#     precision: str = "fp16"  # fp32, fp16, int8
#     batch_size: int = 1
#     max_workspace_size: int = 1 << 30  # 1GB
#     use_dynamic_shapes: bool = True
#     optimization_level: int = 3
#     calibration_cache: Optional[str] = None
#     engine_cache_dir: Optional[str] = None
#
#
# class NeRFCalibrationDataloader:
#     """Custom calibration dataloader for NeRF models"""
#
#     def __init__(self, dataset, batch_size: int = 32, max_samples: int = 1000):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.max_samples = max_samples
#         self.current_idx = 0
#
#     def __iter__(self):
#         self.current_idx = 0
#         return self
#
#     def __next__(self):
#         if self.current_idx >= min(len(self.dataset), self.max_samples):
#             raise StopIteration
#
#         # Get batch of data
#         batch_data = []
#         for i in range(self.batch_size):
#             if self.current_idx + i >= len(self.dataset):
#                 break
#             batch_data.append(self.dataset[self.current_idx + i])
#
#         self.current_idx += len(batch_data)
#
#         if not batch_data:
#             raise StopIteration
#
#         # Convert to tensors (adapt based on your dataset format)
#         rays = torch.stack([item["rays"] for item in batch_data])
#         if "ts" in batch_data[0]:
#             ts = torch.stack([item["ts"] for item in batch_data])
#             return {"rays": rays, "ts": ts}
#         else:
#             return {"rays": rays, "ts": None}
#
#
# class QuantizationProfiler:
#     """Profiler for quantized NeRF models"""
#
#     def __init__(self, model, device: str = "cuda:0"):
#         self.model = model
#         self.device = device
#         self.results = {}
#
#     def profile_inference(self, rays: torch.Tensor, ts: Optional[torch.Tensor] = None,
#                           num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, float]:
#         """Profile inference performance"""
#         self.model.eval()
#         self.model.to(self.device)
#
#         rays = rays.to(self.device)
#         if ts is not None:
#             ts = ts.to(self.device)
#
#         # Warmup
#         with torch.no_grad():
#             for _ in range(warmup_runs):
#                 _ = self.model(rays, ts)
#
#         # Synchronize GPU
#         if self.device.startswith("cuda"):
#             torch.cuda.synchronize()
#
#         # Actual timing
#         times = []
#         with torch.no_grad():
#             for _ in range(num_runs):
#                 start_time = time.perf_counter()
#
#                 if self.device.startswith("cuda"):
#                     torch.cuda.synchronize()
#
#                 _ = self.model(rays, ts)
#
#                 if self.device.startswith("cuda"):
#                     torch.cuda.synchronize()
#
#                 end_time = time.perf_counter()
#                 times.append((end_time - start_time) * 1000)  # Convert to ms
#
#         results = {
#             "mean_time_ms": np.mean(times),
#             "std_time_ms": np.std(times),
#             "min_time_ms": np.min(times),
#             "max_time_ms": np.max(times),
#             "throughput_fps": 1000.0 / np.mean(times) if np.mean(times) > 0 else 0
#         }
#
#         self.results["inference"] = results
#         return results
#
#     def profile_memory(self, rays: torch.Tensor, ts: Optional[torch.Tensor] = None) -> Dict[str, int]:
#         """Profile memory usage"""
#         if not self.device.startswith("cuda"):
#             return {"error": "Memory profiling only available on CUDA"}
#
#         self.model.eval()
#         self.model.to(self.device)
#
#         rays = rays.to(self.device)
#         if ts is not None:
#             ts = ts.to(self.device)
#
#         # Clear cache and measure baseline
#         torch.cuda.empty_cache()
#         torch.cuda.reset_peak_memory_stats()
#         baseline_memory = torch.cuda.memory_allocated()
#
#         # Run inference
#         with torch.no_grad():
#             _ = self.model(rays, ts)
#
#         peak_memory = torch.cuda.max_memory_allocated()
#         current_memory = torch.cuda.memory_allocated()
#
#         results = {
#             "baseline_memory_mb": baseline_memory / 1024 / 1024,
#             "peak_memory_mb": peak_memory / 1024 / 1024,
#             "current_memory_mb": current_memory / 1024 / 1024,
#             "memory_increase_mb": (peak_memory - baseline_memory) / 1024 / 1024
#         }
#
#         self.results["memory"] = results
#         return results
#
#     def compare_models(self, original_model, quantized_model,
#                        rays: torch.Tensor, ts: Optional[torch.Tensor] = None) -> Dict[str, any]:
#         """Compare original vs quantized model performance"""
#
#         # Profile original model
#         original_profiler = QuantizationProfiler(original_model, self.device)
#         orig_inference = original_profiler.profile_inference(rays, ts)
#         orig_memory = original_profiler.profile_memory(rays, ts)
#
#         # Profile quantized model
#         quant_inference = self.profile_inference(rays, ts)
#         quant_memory = self.profile_memory(rays, ts)
#
#         # Calculate improvements
#         speedup = orig_inference["mean_time_ms"] / quant_inference["mean_time_ms"]
#         memory_reduction = (orig_memory["peak_memory_mb"] - quant_memory["peak_memory_mb"]) / orig_memory[
#             "peak_memory_mb"] * 100
#
#         comparison = {
#             "original": {
#                 "inference": orig_inference,
#                 "memory": orig_memory
#             },
#             "quantized": {
#                 "inference": quant_inference,
#                 "memory": quant_memory
#             },
#             "improvements": {
#                 "speedup_factor": speedup,
#                 "memory_reduction_percent": memory_reduction,
#                 "throughput_improvement": quant_inference["throughput_fps"] - orig_inference["throughput_fps"]
#             }
#         }
#
#         return comparison
#
#
# class TensorRTDeployment:
#     """TensorRT deployment utilities for NeRF models"""
#
#     def __init__(self, config: DeploymentConfig):
#         self.config = config
#         self.engine = None
#         self.context = None
#
#     def build_engine_from_onnx(self, onnx_path: str, engine_path: str) -> bool:
#         """Build TensorRT engine from ONNX model"""
#         if not TENSORRT_AVAILABLE:
#             logger.error("TensorRT not available")
#             return False
#
#         try:
#             # Create builder and config
#             TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#             builder = trt.Builder(TRT_LOGGER)
#             config = builder.create_builder_config()
#
#             # Set workspace size
#             config.max_workspace_size = self.config.max_workspace_size
#
#             # Set precision
#             if self.config.precision == "fp16":
#                 config.set_flag(trt.BuilderFlag.FP16)
#             elif self.config.precision == "int8":
#                 config.set_flag(trt.BuilderFlag.INT8)
#                 if self.config.calibration_cache:
#                     config.int8_calibrator = self._create_calibrator()
#
#             # Parse ONNX
#             network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
#             parser = trt.OnnxParser(network, TRT_LOGGER)
#
#             with open(onnx_path, 'rb') as model:
#                 if not parser.parse(model.read()):
#                     logger.error("Failed to parse ONNX model")
#                     for error in range(parser.num_errors):
#                         logger.error(parser.get_error(error))
#                     return False
#
#             # Configure dynamic shapes if enabled
#             if self.config.use_dynamic_shapes:
#                 self._configure_dynamic_shapes(config, network)
#
#             # Build engine
#             logger.info("Building TensorRT engine... This may take a while.")
#             engine = builder.build_engine(network, config)
#
#             if engine is None:
#                 logger.error("Failed to build TensorRT engine")
#                 return False
#
#             # Save engine
#             with open(engine_path, 'wb') as f:
#                 f.write(engine.serialize())
#
#             logger.info(f"TensorRT engine saved to {engine_path}")
#             return True
#
#         except Exception as e:
#             logger.error(f"Error building TensorRT engine: {e}")
#             return False
#
#     def _configure_dynamic_shapes(self, config, network):
#         """Configure dynamic input shapes for TensorRT"""
#         profile = config.create_optimization_profile()
#
#         # Configure rays input (batch_size can vary)
#         rays_input = network.get_input(0)
#         profile.set_shape(rays_input.name,
#                           (1, rays_input.shape[1]),  # min
#                           (1024, rays_input.shape[1]),  # opt
#                           (8192, rays_input.shape[1]))  # max
#
#         # Configure timestamps input if present
#         if network.num_inputs > 1:
#             ts_input = network.get_input(1)
#             profile.set_shape(ts_input.name,
#                               (1,),  # min
#                               (1024,),  # opt
#                               (8192,))  # max
#
#         config.add_optimization_profile(profile)
#
#     def _create_calibrator(self):
#         """Create INT8 calibrator"""
#         # This would need to be implemented based on your calibration dataset
#         # For now, return None (would need custom calibrator class)
#         logger.warning("INT8 calibration not implemented - using FP16")
#         return None
#
#     def load_engine(self, engine_path: str) -> bool:
#         """Load TensorRT engine from file"""
#         if not TENSORRT_AVAILABLE:
#             logger.error("TensorRT not available")
#             return False
#
#         try:
#             TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#             runtime = trt.Runtime(TRT_LOGGER)
#
#             with open(engine_path, 'rb') as f:
#                 self.engine = runtime.deserialize_cuda_engine(f.read())
#
#             if self.engine is None:
#                 logger.error("Failed to load TensorRT engine")
#                 return False
#
#             self.context = self.engine.create_execution_context()
#             logger.info(f"Loaded TensorRT engine from {engine_path}")
#             return True
#
#         except Exception as e:
#             logger.error(f"Error loading TensorRT engine: {e}")
#             return False
#
#     def infer(self, rays: np.ndarray, ts: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
#         """Run inference with TensorRT engine"""
#         if self.engine is None or self.context is None:
#             raise RuntimeError("TensorRT engine not loaded")
#
#         # Allocate GPU memory
#         import pycuda.driver as cuda
#         import pycuda.autoinit
#
#         # Set input shapes
#         batch_size = rays.shape[0]
#         self.context.set_binding_shape(0, rays.shape)
#         if ts is not None:
#             self.context.set_binding_shape(1, ts.shape)
#
#         # Allocate memory
#         inputs, outputs, bindings, stream = self._allocate_buffers()
#
#         # Copy input data
#         np.copyto(inputs[0].host, rays.ravel())
#         if ts is not None and len(inputs) > 1:
#             np.copyto(inputs[1].host, ts.ravel())
#
#         # Run inference
#         self._do_inference(self.context, bindings, inputs, outputs, stream)
#
#         # Process outputs
#         results = {}
#         output_names = ["rgb", "depth", "weights"]  # Adjust based on your model
#         for i, name in enumerate(output_names[:len(outputs)]):
#             if i < len(outputs):
#                 results[name] = outputs[i].host.reshape(batch_size, -1)
#
#         return results
#
#     def _allocate_buffers(self):
#         """Allocate GPU buffers for inference"""
#         import pycuda.driver as cuda
#
#         inputs = []
#         outputs = []
#         bindings = []
#         stream = cuda.Stream()
#
#         for binding in self.engine:
#             binding_idx = self.engine.get_binding_index(binding)
#             size = trt.volume(self.context.get_binding_shape(binding_idx))
#             dtype = trt.nptype(self.engine.get_binding_dtype(binding))
#
#             # Allocate host and device buffers
#             host_mem = cuda.pagelocked_empty(size, dtype)
#             device_mem = cuda.mem_alloc(host_mem.nbytes)
#
#             bindings.append(int(device_mem))
#
#             if self.engine.binding_is_input(binding):
#                 inputs.append(self._HostDeviceMem(host_mem, device_mem))
#             else:
#                 outputs.append(self._HostDeviceMem(host_mem, device_mem))
#
#         return inputs, outputs, bindings, stream
#
#     def _do_inference(self, context, bindings, inputs, outputs, stream):
#         """Execute inference"""
#         import pycuda.driver as cuda
#
#         # Copy inputs to GPU
#         [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
#
#         # Execute
#         context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
#
#         # Copy outputs back
#         [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
#
#         # Synchronize
#         stream.synchronize()
#
#     class _HostDeviceMem:
#         """Helper class for managing host/device memory pairs"""
#
#         def __init__(self, host_mem, device_mem):
#             self.host = host_mem
#             self.device = device_mem
#
#
# class PolygraphyOptimizer:
#     """Polygraphy-based optimization and comparison utilities"""
#
#     def __init__(self):
#         if not POLYGRAPHY_AVAILABLE:
#             raise RuntimeError("Polygraphy not available")
#
#     def compare_precision_modes(self, onnx_path: str, output_dir: str) -> Dict[str, Dict]:
#         """Compare different precision modes using Polygraphy"""
#         results = {}
#         precisions = ["fp32", "fp16", "int8"]
#
#         for precision in precisions:
#             try:
#                 # Create TensorRT config for this precision
#                 profiles = [
#                     Profile().add("rays", (1, 11), (1024, 11), (8192, 11))
#                 ]
#
#                 config = CreateConfig(
#                     max_workspace_size=1 << 30,
#                     profiles=profiles,
#                     fp16=(precision == "fp16"),
#                     int8=(precision == "int8")
#                 )
#
#                 # Build and profile engine
#                 engine_path = f"{output_dir}/engine_{precision}.trt"
#                 self._build_and_profile_engine(onnx_path, engine_path, config)
#
#                 results[precision] = {
#                     "engine_path": engine_path,
#                     "build_success": True
#                 }
#
#             except Exception as e:
#                 logger.error(f"Failed to build {precision} engine: {e}")
#                 results[precision] = {
#                     "engine_path": None,
#                     "build_success": False,
#                     "error": str(e)
#                 }
#
#         return results
#
#     def _build_and_profile_engine(self, onnx_path: str, engine_path: str, config):
#         """Build and profile TensorRT engine using Polygraphy"""
#         # Load ONNX model
#         model = OnnxFromPath(onnx_path)
#
#         # Build TensorRT engine
#         engine = EngineFromNetwork(model, config=config)
#
#         # Save engine
#         with engine as eng:
#             with open(engine_path, 'wb') as f:
#                 f.write(eng.serialize())
#
#     def accuracy_comparison(self, onnx_path: str, engines: Dict[str, str],
#                             test_data: List[Dict]) -> Dict[str, Dict]:
#         """Compare accuracy between different precision engines"""
#         results = {}
#
#         # Get reference outputs from ONNX model
#         onnx_runner = mod.OnnxrtRunner(OnnxFromPath(onnx_path))
#
#         for precision, engine_path in engines.items():
#             if not os.path.exists(engine_path):
#                 continue
#
#             try:
#                 # Create TensorRT runner
#                 trt_runner = TrtRunner(EngineFromNetwork(engine_path))
#
#                 # Compare outputs
#                 accuracy_metrics = self._compute_accuracy_metrics(
#                     onnx_runner, trt_runner, test_data
#                 )
#
#                 results[precision] = accuracy_metrics
#
#             except Exception as e:
#                 logger.error(f"Accuracy comparison failed for {precision}: {e}")
#                 results[precision] = {"error": str(e)}
#
#         return results
#
#     def _compute_accuracy_metrics(self, reference_runner, test_runner, test_data):
#         """Compute accuracy metrics between two runners"""
#         mse_values = []
#         psnr_values = []
#
#         for data in test_data[:10]:  # Limit to first 10 samples
#             # Run reference
#             ref_outputs = reference_runner.infer(data)
#
#             # Run test
#             test_outputs = test_runner.infer(data)
#
#             # Compare RGB outputs
#             if "rgb" in ref_outputs and "rgb" in test_outputs:
#                 ref_rgb = ref_outputs["rgb"]
#                 test_rgb = test_outputs["rgb"]
#
#                 mse = np.mean((ref_rgb - test_rgb) ** 2)
#                 psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
#
#                 mse_values.append(mse)
#                 psnr_values.append(psnr)
#
#         return {
#             "mse_mean": np.mean(mse_values),
#             "mse_std": np.std(mse_values),
#             "psnr_mean": np.mean(psnr_values),
#             "psnr_std": np.std(psnr_values),
#             "num_samples": len(mse_values)
#         }
#
#
# class NeRFDeploymentPipeline:
#     """Complete deployment pipeline for NeRF models"""
#
#     def __init__(self, config: DeploymentConfig):
#         self.config = config
#         self.profiler = None
#         self.trt_deployment = TensorRTDeployment(config)
#         self.polygraphy_optimizer = None
#
#         if POLYGRAPHY_AVAILABLE:
#             self.polygraphy_optimizer = PolygraphyOptimizer()
#
#     def deploy_model(self, model_path: str, output_dir: str,
#                      calibration_dataset=None) -> Dict[str, any]:
#         """Complete deployment pipeline"""
#         os.makedirs(output_dir, exist_ok=True)
#
#         results = {
#             "input_model": model_path,
#             "output_directory": output_dir,
#             "steps_completed": [],
#             "performance_metrics": {},
#             "deployment_artifacts": {}
#         }
#
#         try:
#             # Step 1: Convert to ONNX if needed
#             onnx_path = self._ensure_onnx_format(model_path, output_dir)
#             results["steps_completed"].append("onnx_conversion")
#             results["deployment_artifacts"]["onnx_model"] = onnx_path
#
#             # Step 2: Optimize with Polygraphy if available
#             if self.polygraphy_optimizer:
#                 optimization_results = self._optimize_with_polygraphy(onnx_path, output_dir)
#                 results["steps_completed"].append("polygraphy_optimization")
#                 results["performance_metrics"]["precision_comparison"] = optimization_results
#
#             # Step 3: Build TensorRT engines
#             engine_results = self._build_tensorrt_engines(onnx_path, output_dir)
#             results["steps_completed"].append("tensorrt_build")
#             results["deployment_artifacts"]["engines"] = engine_results
#
#             # Step 4: Performance benchmarking
#             if calibration_dataset:
#                 benchmark_results = self._benchmark_engines(engine_results, calibration_dataset)
#                 results["steps_completed"].append("performance_benchmark")
#                 results["performance_metrics"]["benchmarks"] = benchmark_results
#
#             # Step 5: Generate deployment scripts
#             deployment_scripts = self._generate_deployment_scripts(output_dir, engine_results)
#             results["steps_completed"].append("deployment_scripts")
#             results["deployment_artifacts"]["scripts"] = deployment_scripts
#
#             logger.info("Deployment pipeline completed successfully!")
#
#         except Exception as e:
#             logger.error(f"Deployment pipeline failed: {e}")
#             results["error"] = str(e)
#
#         # Save results
#         with open(f"{output_dir}/deployment_report.json", 'w') as f:
#             json.dump(results, f, indent=2)
#
#         return results
#
#     def _ensure_onnx_format(self, model_path: str, output_dir: str) -> str:
#         """Ensure model is in ONNX format"""
#         if model_path.endswith('.onnx'):
#             return model_path
#
#         # Convert PyTorch model to ONNX
#         onnx_path = f"{output_dir}/model.onnx"
#
#         # Load PyTorch model
#         model = torch.load(model_path, map_location='cpu')
#         model.eval()
#
#         # Create dummy inputs
#         dummy_rays = torch.randn(1, 11)
#         dummy_ts = torch.randint(0, 10, (1,))
#
#         # Export to ONNX
#         torch.onnx.export(
#             model,
#             (dummy_rays, dummy_ts),
#             onnx_path,
#             input_names=['rays', 'timestamps'],
#             output_names=['rgb', 'depth', 'weights'],
#             dynamic_axes={
#                 'rays': {0: 'batch_size'},
#                 'timestamps': {0: 'batch_size'},
#                 'rgb': {0: 'batch_size'},
#                 'depth': {0: 'batch_size'},
#                 'weights': {0: 'batch_size'}
#             },
#             opset_version=11
#         )
#
#         return onnx_path
#
#     def _optimize_with_polygraphy(self, onnx_path: str, output_dir: str) -> Dict:
#         """Optimize model using Polygraphy"""
#         if not self.polygraphy_optimizer:
#             return {"error": "Polygraphy not available"}
#
#         poly_output_dir = f"{output_dir}/polygraphy_analysis"
#         os.makedirs(poly_output_dir, exist_ok=True)
#
#         # Compare precision modes
#         precision_results = self.polygraphy_optimizer.compare_precision_modes(
#             onnx_path, poly_output_dir
#         )
#
#         return precision_results
#
#     def _build_tensorrt_engines(self, onnx_path: str, output_dir: str) -> Dict[str, str]:
#         """Build TensorRT engines for different precision modes"""
#         engines = {}
#
#         for precision in ["fp32", "fp16"]:  # Skip int8 for now due to calibration complexity
#             try:
#                 engine_path = f"{output_dir}/nerf_engine_{precision}.trt"
#
#                 # Update config for this precision
#                 config = DeploymentConfig(
#                     precision=precision,
#                     batch_size=self.config.batch_size,
#                     max_workspace_size=self.config.max_workspace_size,
#                     use_dynamic_shapes=self.config.use_dynamic_shapes
#                 )
#
#                 trt_builder = TensorRTDeployment(config)
#
#                 if trt_builder.build_engine_from_onnx(onnx_path, engine_path):
#                     engines[precision] = engine_path
#                     logger.info(f"Built {precision} TensorRT engine: {engine_path}")
#                 else:
#                     logger.warning(f"Failed to build {precision} TensorRT engine")
#
#             except Exception as e:
#                 logger.error(f"Error building {precision} engine: {e}")
#
#         return engines
#
#     def _benchmark_engines(self, engines: Dict[str, str], calibration_dataset) -> Dict:
#         """Benchmark TensorRT engines"""
#         benchmark_results = {}
#
#         # Create sample data for benchmarking
#         sample_rays = np.random.randn(1024, 11).astype(np.float32)
#         sample_ts = np.random.randint(0, 10, (1024,)).astype(np.int32)
#
#         for precision, engine_path in engines.items():
#             try:
#                 # Load engine
#                 trt_deploy = TensorRTDeployment(self.config)
#                 if not trt_deploy.load_engine(engine_path):
#                     continue
#
#                 # Benchmark inference
#                 times = []
#                 for _ in range(50):  # 50 runs
#                     start_time = time.perf_counter()
#                     _ = trt_deploy.infer(sample_rays, sample_ts)
#                     end_time = time.perf_counter()
#                     times.append((end_time - start_time) * 1000)
#
#                 benchmark_results[precision] = {
#                     "mean_time_ms": np.mean(times),
#                     "std_time_ms": np.std(times),
#                     "min_time_ms": np.min(times),
#                     "max_time_ms": np.max(times),
#                     "throughput_fps": 1000.0 / np.mean(times)
#                 }
#
#             except Exception as e:
#                 logger.error(f"Benchmarking failed for {precision}: {e}")
#                 benchmark_results[precision] = {"error": str(e)}
#
#         return benchmark_results
#
#     def _generate_deployment_scripts(self, output_dir: str, engines: Dict[str, str]) -> Dict[str, str]:
#         """Generate deployment scripts and utilities"""
#         scripts = {}
#
#         # Python inference script
#         inference_script = self._create_inference_script(engines)
#         inference_path = f"{output_dir}/inference.py"
#         with open(inference_path, 'w') as f:
#             f.write(inference_script)
#         scripts["inference_script"] = inference_path
#
#         # Docker deployment script
#         docker_script = self._create_docker_script()
#         docker_path = f"{output_dir}/Dockerfile"
#         with open(docker_path, 'w') as f:
#             f.write(docker_script)
#         scripts["dockerfile"] = docker_path
#
#         # Deployment README
#         readme_content = self._create_deployment_readme(engines)
#         readme_path = f"{output_dir}/README.md"
#         with open(readme_path, 'w') as f:
#             f.write(readme_content)
#         scripts["readme"] = readme_path
#
#         return scripts
#
#     def _create_inference_script(self, engines: Dict[str, str]) -> str:
#         """Create Python inference script"""
#         return f'''#!/usr/bin/env python3
# """
# TensorRT Inference Script for NeRF Models
# Generated by NeRF QAT Deployment Pipeline
# """
#
# import numpy as np
# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit
# from typing import Dict, Optional
# import argparse
# import time
#
# class NeRFTensorRTInference:
#     def __init__(self, engine_path: str):
#         self.engine_path = engine_path
#         self.engine = None
#         self.context = None
#         self._load_engine()
#
#     def _load_engine(self):
#         TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#         runtime = trt.Runtime(TRT_LOGGER)
#
#         with open(self.engine_path, 'rb') as f:
#             self.engine = runtime.deserialize_cuda_engine(f.read())
#
#         self.context = self.engine.create_execution_context()
#
#     def infer(self, rays: np.ndarray, ts: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
#         # Set input