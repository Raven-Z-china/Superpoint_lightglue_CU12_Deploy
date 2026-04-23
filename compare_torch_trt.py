#!/usr/bin/env python3
"""
Compare custom SuperPoint/LightGlue (torch) vs official TensorRT Engine outputs.
Uses real images from freiburg_sequence to test the full pipeline.
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import time
import sys
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from scipy.ndimage import maximum_filter
from pathlib import Path


def load_tensorrt_engine(engine_path):
    """Load TensorRT engine and create context."""
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()
    return engine, context


def allocate_trt_buffers(engine, context, input_shapes, output_shapes):
    """Allocate GPU buffers for TensorRT engine."""
    bindings = []
    d_inputs = {}
    io_tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    
    for name in io_tensor_names:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            shape = input_shapes.get(name)
            if shape is None:
                shape = context.get_tensor_shape(name)
            context.set_input_shape(name, shape)
            shape = context.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            # Handle TensorRT Dims object
            if hasattr(shape, '__len__'):
                size = int(np.prod([int(d) for d in shape]))
            else:
                size = int(shape)
            d_input = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            d_inputs[name] = (d_input, shape, dtype)
            bindings.append(int(d_input))
        else:
            shape = context.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            if hasattr(shape, '__len__'):
                size = int(np.prod([int(d) for d in shape]))
            else:
                size = int(shape)
            d_output = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            d_inputs[name] = (d_output, shape, dtype)
            bindings.append(int(d_output))
    
    return d_inputs, bindings


def set_trt_inputs(context, d_inputs, inputs_dict):
    """Copy data to GPU buffers."""
    for name, data in inputs_dict.items():
        if isinstance(data, np.ndarray) and name in d_inputs:
            cuda.memcpy_htod(int(d_inputs[name][0]), data)


def run_trt(context, bindings, d_inputs):
    """Run TensorRT inference."""
    context.execute_v2(bindings)
    
    outputs = {}
    for name, (d_ptr, shape, dtype) in d_inputs.items():
        if len(bindings) == 0:
            continue
        h_output = np.empty(shape, dtype=dtype)
        cuda.memcpy_dtoh(h_output, int(d_ptr))
        outputs[name] = h_output
    return outputs


def load_image(img_path, target_size=None):
    """Load and preprocess image."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {img_path}")
    
    original_shape = img.shape
    if target_size:
        img = cv2.resize(img, target_size)
    
    img = img.astype(np.float32) / 255.0
    return img, original_shape


def prepare_image_for_superpoint(img, target_h=480, target_w=640):
    """Prepare image for SuperPoint input."""
    img_resized = cv2.resize(img, (target_w, target_h))
    return img_resized.reshape(1, 1, target_h, target_w)


def extract_keypoints_nms(scores, nms_radius=4, threshold=0.005):
    """Extract keypoints using NMS from dense scores."""
    # Handle both 3D (1, H, W) and 4D (1, 1, H, W) inputs
    if scores.dim() == 4:
        scores_flat = scores[0, 0].cpu().numpy()
        H, W = scores.shape[2], scores.shape[3]
    else:  # 3D
        scores_flat = scores[0].cpu().numpy()
        H, W = scores.shape[1], scores.shape[2]
    
    local_max = maximum_filter(scores_flat, size=nms_radius)
    peaks = (scores_flat == local_max) & (scores_flat > threshold)
    
    ys, xs = np.where(peaks)
    scores_list = scores_flat[ys, xs]
    
    indices = np.argsort(-scores_list)
    keypoints = np.stack([xs[indices], ys[indices]], axis=-1).astype(np.float32)
    scores_out = scores_list[indices]
    
    return keypoints, scores_out


def sample_descriptors_torch(keypoints, descriptors, s=8):
    """Sample descriptors at keypoint locations (ONNX-compatible version)."""
    b, c, h, w = descriptors.shape
    device = descriptors.device
    
    keypoints = keypoints - s / 2 + 0.5
    scale = torch.tensor([w * s - s / 2 - 0.5, h * s - s / 2 - 0.5], 
                         device=device, dtype=keypoints.dtype)
    keypoints = keypoints / scale[None]
    keypoints = keypoints * 2 - 1
    
    b_kp, num_kp, _ = keypoints.shape
    keypoints_grid = keypoints.view(b_kp, 1, num_kp, 2)
    
    desc_sampled = F.grid_sample(
        descriptors, keypoints_grid, mode="bilinear", align_corners=True
    )
    desc_sampled = desc_sampled.reshape(b, c, -1).permute(0, 2, 1)
    desc_sampled = F.normalize(desc_sampled, p=2.0, dim=-1)
    
    return desc_sampled


def normalize_keypoints_torch(kpts, size=None):
    """Normalize keypoints to [-1, 1] range."""
    if size is None:
        size = 1 + kpts.max(-2).values - kpts.min(-2).values
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)[None]
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts


def compare_outputs(name, torch_out, trt_out, rtol=1e-2, atol=1e-2):
    """Compare custom torch and TensorRT outputs."""
    torch_arr = torch_out.detach().cpu().numpy().astype(np.float32)
    if isinstance(trt_out, torch.Tensor):
        trt_arr = trt_out.detach().cpu().numpy().astype(np.float32)
    else:
        trt_arr = trt_out.astype(np.float32)
    
    min_shape = min(torch_arr.shape[-2:], trt_arr.shape[-2:])
    torch_cmp = torch_arr[..., :min_shape[0], :min_shape[1]]
    trt_cmp = trt_arr[..., :min_shape[0], :min_shape[1]]
    
    abs_diff = np.abs(torch_cmp - trt_cmp)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    
    denom = np.maximum(np.abs(torch_cmp), 1e-8)
    rel_error = np.abs(torch_cmp - trt_cmp) / denom
    max_rel = np.max(rel_error)
    
    match = np.allclose(torch_cmp, trt_cmp, rtol=rtol, atol=atol)
    
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Torch shape: {torch_arr.shape}, TRT shape: {trt_arr.shape}")
    print(f"Max absolute diff: {max_diff:.6e}")
    print(f"Mean absolute diff: {mean_diff:.6e}")
    print(f"Max relative error: {max_rel:.6e}")
    print(f"Match (rtol={rtol}, atol={atol}): {'PASS' if match else 'FAIL'}")
    
    return match, max_diff, mean_diff


def get_file_names(directory):
    """Get all image file names from directory."""
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    files = []
    for f in os.listdir(directory):
        ext = os.path.splitext(f)[1].lower()
        if ext in image_extensions:
            files.append(os.path.join(directory, f))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description='Compare custom torch vs official TRT')
    parser.add_argument('--config', '-c', default='config/config.yaml',
                        help='Config file path')
    parser.add_argument('--weights', '-w', default='weights',
                        help='Weights directory')
    parser.add_argument('--images', '-i', default='image/freiburg_sequence',
                        help='Input images directory')
    parser.add_argument('--output', '-o', default='image/freiburg_sequence/match_images',
                        help='Output directory for matches')
    parser.add_argument('--img-h', type=int, default=480, help='Image height')
    parser.add_argument('--img-w', type=int, default=640, help='Image width')
    parser.add_argument('--max-kp', type=int, default=2048, help='Max keypoints')
    parser.add_argument('--iter', type=int, default=10, help='Benchmark iterations')
    parser.add_argument('--warmup', type=int, default=2, help='Warmup iterations')
    parser.add_argument('--nms-radius', type=int, default=4, help='NMS radius')
    parser.add_argument('--kp-threshold', type=float, default=0.005, help='Keypoint threshold')
    args = parser.parse_args()
    
    print("="*60)
    print("Custom Torch vs Official TensorRT Engine Comparison")
    print("="*60)
    
    # Load custom PyTorch models
    sys.path.insert(0, 'convert2onnx')
    from custom_superpoint import SuperPoint as SuperPointTorch
    from custom_lightglue import LightGlue as LightGlueTorch
    
    print("\nLoading custom PyTorch models...")
    sp_torch = SuperPointTorch({'max_keypoints': args.max_kp, 'nms_radius': args.nms_radius})
    lg_torch = LightGlueTorch({'max_keypoints': args.max_kp})
    
    sp_torch = sp_torch.cuda()
    lg_torch = lg_torch.cuda()
    sp_torch.eval()
    lg_torch.eval()
    print("Custom PyTorch models loaded")
    
    # Load official TensorRT engines
    sp_engine_path = os.path.join(args.weights, 'offical_sp_FP32_v1.engine')
    lg_engine_path = os.path.join(args.weights, 'offical_lg_FP32_v1.engine')
    
    print(f"\nLoading TensorRT engines...")
    print(f"  SuperPoint: {sp_engine_path} (FP32)")
    print(f"  LightGlue: {lg_engine_path} (FP32)")
    
    sp_engine, sp_context = load_tensorrt_engine(sp_engine_path)
    lg_engine, lg_context = load_tensorrt_engine(lg_engine_path)
    
    print("TensorRT engines loaded")
    
    # Print engine info
    print("\n--- SuperPoint Engine I/O ---")
    for i in range(sp_engine.num_io_tensors):
        name = sp_engine.get_tensor_name(i)
        mode = sp_engine.get_tensor_mode(name)
        dtype = sp_engine.get_tensor_dtype(name)
        print(f"  {name}: {mode}, {dtype}")
    
    print("\n--- LightGlue Engine I/O ---")
    for i in range(lg_engine.num_io_tensors):
        name = lg_engine.get_tensor_name(i)
        mode = lg_engine.get_tensor_mode(name)
        dtype = lg_engine.get_tensor_dtype(name)
        print(f"  {name}: {mode}, {dtype}")
    
    # Get image list
    image_names = get_file_names(args.images)
    if len(image_names) < 2:
        print(f"Need at least 2 images, found {len(image_names)}")
        return
    
    print(f"\nFound {len(image_names)} images")
    
    # Process first image
    print("\n" + "="*60)
    print("Processing image 0 as reference")
    print("="*60)
    
    img0, shape0 = load_image(image_names[0])
    img0_t = prepare_image_for_superpoint(img0, args.img_h, args.img_w)
    print(f"Image 0: {shape0} -> {args.img_h}x{args.img_w}")
    
    # ============== SUPERPOINT COMPARISON ==============
    print("\n" + "="*60)
    print("STEP 1: SuperPoint Feature Extraction Comparison")
    print("="*60)
    
    # Custom torch inference
    with torch.no_grad():
        img0_cuda = torch.from_numpy(img0_t).cuda()
        scores0_torch, desc0_torch = sp_torch(img0_cuda)
    
    print(f"\nCustom SuperPoint (torch) outputs:")
    print(f"  scores: {scores0_torch.shape}, dtype: {scores0_torch.dtype}")
    print(f"  descriptors: {desc0_torch.shape}, dtype: {desc0_torch.dtype}")
    print(f"  scores range: [{scores0_torch.min():.4f}, {scores0_torch.max():.4f}]")
    
    # Official TensorRT inference
    # Prepare inputs first to set shapes
    sp_inputs = {'image': img0_t.astype(np.float32)}
    sp_input_shapes = {'image': img0_t.shape}
    sp_d_inputs, sp_bindings = allocate_trt_buffers(sp_engine, sp_context, sp_input_shapes, {})
    set_trt_inputs(sp_context, sp_d_inputs, sp_inputs)
    sp_context.execute_v2(sp_bindings)
    sp_trt_outputs = {}
    for name, (d_ptr, shape, dtype) in sp_d_inputs.items():
        if sp_engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            h_output = np.empty(shape, dtype=dtype)
            cuda.memcpy_dtoh(h_output, int(d_ptr))
            sp_trt_outputs[name] = h_output
    
    for name in sp_trt_outputs:
        print(f"\nOfficial SuperPoint (TRT) output: {name}, shape: {sp_trt_outputs[name].shape}")
    
    # Get TRT outputs
    trt_scores = None
    trt_desc = None
    for name in sp_trt_outputs:
        if 'scores' in name.lower():
            trt_scores = sp_trt_outputs[name]
        elif 'descriptor' in name.lower():
            trt_desc = sp_trt_outputs[name]
    
    # Compare dense scores - Both should be (1, 480, 640) now
    if trt_scores is not None:
        trt_scores_tensor = torch.from_numpy(trt_scores).cuda().squeeze(1)  # (1, 480, 640)
        print(f"\n--- Direct dense output comparison ---")
        print(f"Torch scores shape: {scores0_torch.shape}")
        print(f"TRT scores shape: {trt_scores_tensor.shape}")
        
        compare_outputs("SuperPoint Scores", scores0_torch, 
                        trt_scores_tensor, rtol=1e-1, atol=1e-1)
    
    if trt_desc is not None:
        trt_desc_tensor = torch.from_numpy(trt_desc).cuda()
        print(f"\n--- Descriptor comparison ---")
        print(f"Torch desc shape: {desc0_torch.shape}")
        print(f"TRT desc shape: {trt_desc_tensor.shape}")
        
        compare_outputs("SuperPoint Descriptors (dense)", desc0_torch,
                        trt_desc_tensor, rtol=1e-2, atol=1e-2)
    
    # Extract keypoints from custom model outputs
    kp0_torch, scores_kp0 = extract_keypoints_nms(
        scores0_torch, nms_radius=args.nms_radius, threshold=args.kp_threshold
    )
    
    # Sample descriptors for keypoints
    desc0_sampled_torch = sample_descriptors_torch(
        torch.from_numpy(kp0_torch)[None].cuda(), desc0_torch
    )
    
    print(f"\nExtracted keypoints from custom model:")
    print(f"  keypoints: {kp0_torch.shape}")
    print(f"  descriptors: {desc0_sampled_torch.shape}")
    
    # Process second image
    print("\n" + "="*60)
    print("Processing image 1 for LightGlue comparison")
    print("="*60)
    
    img1, shape1 = load_image(image_names[1])
    img1_t = prepare_image_for_superpoint(img1, args.img_h, args.img_w)
    
    with torch.no_grad():
        img1_cuda = torch.from_numpy(img1_t).cuda()
        scores1_torch, desc1_torch = sp_torch(img1_cuda)
    
    kp1_torch, scores_kp1 = extract_keypoints_nms(
        scores1_torch, nms_radius=args.nms_radius, threshold=args.kp_threshold
    )
    desc1_sampled_torch = sample_descriptors_torch(
        torch.from_numpy(kp1_torch)[None].cuda(), desc1_torch
    )
    
    print(f"\nImage 1 keypoints: {kp1_torch.shape}")
    
    # ============== LIGHTGLUE COMPARISON ==============
    print("\n" + "="*60)
    print("STEP 2: LightGlue Feature Matching Comparison")
    print("="*60)
    
    # Ensure keypoints are padded to same length for LightGlue
    max_kpts = max(len(kp0_torch), len(kp1_torch))
    print(f"\nKeypoint counts: img0={len(kp0_torch)}, img1={len(kp1_torch)}, max={max_kpts}")
    
    # Pad keypoints and descriptors to max_kpts
    def pad_to_size(arr, size, dim=0):
        if arr.shape[dim] >= size:
            return arr[:size]
        pad_shape = list(arr.shape)
        pad_shape[dim] = size - arr.shape[dim]
        if dim == 0:
            return torch.cat([arr, torch.zeros(pad_shape, device=arr.device, dtype=arr.dtype)], dim=dim)
        return torch.cat([arr, arr.new_zeros(pad_shape)], dim=dim)
    
    kp0_lg = torch.from_numpy(kp0_torch)[None].cuda()
    kp1_lg = torch.from_numpy(kp1_torch)[None].cuda()
    desc0_lg = desc0_sampled_torch
    desc1_lg = desc1_sampled_torch
    
    # Pad to same size
    if len(kp0_torch) != len(kp1_torch):
        if len(kp0_torch) < len(kp1_torch):
            kp0_lg = pad_to_size(kp0_lg, len(kp1_torch), dim=1)
            desc0_lg = pad_to_size(desc0_lg, len(kp1_torch), dim=1)
        else:
            kp1_lg = pad_to_size(kp1_lg, len(kp0_torch), dim=1)
            desc1_lg = pad_to_size(desc1_lg, len(kp0_torch), dim=1)
    
    print(f"Padded shapes: kp0={kp0_lg.shape}, kp1={kp1_lg.shape}")
    print(f"Padded shapes: desc0={desc0_lg.shape}, desc1={desc1_lg.shape}")
    
    # Custom LightGlue (torch) inference
    with torch.amp.autocast('cuda'):
        scores_lg_torch = lg_torch(
            kp0_lg,
            kp1_lg,
            desc0_lg,
            desc1_lg
        )
    
    print(f"\nCustom LightGlue (torch) output:")
    print(f"  scores: {scores_lg_torch.shape}")
    print(f"  scores range: [{scores_lg_torch.min():.4f}, {scores_lg_torch.max():.4f}]")
    
    # Try TRT LightGlue inference
    print("\nTrying official LightGlue (TRT) inference...")
    try:
        # Normalize keypoints for TRT engine using padded data
        kp0_norm = normalize_keypoints_torch(kp0_lg)
        kp1_norm = normalize_keypoints_torch(kp1_lg)
        
        lg_inputs_trt = {
            'keypoints0': kp0_norm.cpu().numpy().astype(np.float32),
            'keypoints1': kp1_norm.cpu().numpy().astype(np.float32),
            'descriptors0': desc0_lg.cpu().numpy().astype(np.float32),
            'descriptors1': desc1_lg.cpu().numpy().astype(np.float32)
        }
        lg_input_shapes = {
            'keypoints0': lg_inputs_trt['keypoints0'].shape,
            'keypoints1': lg_inputs_trt['keypoints1'].shape,
            'descriptors0': lg_inputs_trt['descriptors0'].shape,
            'descriptors1': lg_inputs_trt['descriptors1'].shape,
        }
        
        lg_d_inputs, lg_bindings = allocate_trt_buffers(lg_engine, lg_context, lg_input_shapes, {})
        set_trt_inputs(lg_context, lg_d_inputs, lg_inputs_trt)
        lg_context.execute_v2(lg_bindings)
        
        lg_trt_outputs = {}
        for name, (d_ptr, shape, dtype) in lg_d_inputs.items():
            if lg_engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                h_output = np.empty([int(d) for d in shape], dtype=dtype)
                cuda.memcpy_dtoh(h_output, int(d_ptr))
                lg_trt_outputs[name] = h_output
        
        for name in lg_trt_outputs:
            print(f"Official LightGlue (TRT) output: {name}, shape: {lg_trt_outputs[name].shape}")
        
        trt_lg_scores = lg_trt_outputs.get('scores')
        if trt_lg_scores is not None:
            compare_outputs("LightGlue Scores", scores_lg_torch,
                            torch.from_numpy(trt_lg_scores).cuda(), rtol=1e-1, atol=1e-1)
    except Exception as e:
        print(f"TRT LightGlue inference skipped: {e}")
        print("  (This is expected if engine was built with different keypoint counts)")
    
    # ============== SEQUENCE COMPARISON ==============
    print("\n" + "="*60)
    print("STEP 3: Sequence Processing Comparison")
    print("="*60)
    
    # Process sequence with custom torch
    print("\nProcessing sequence with custom torch models...")
    
    # Process first image
    kps_all = []
    descs_all = []
    matches_all = []
    
    img0_seq, _ = load_image(image_names[0])
    img0_seq_t = prepare_image_for_superpoint(img0_seq, args.img_h, args.img_w)
    
    with torch.no_grad():
        scores0_seq, desc0_seq = sp_torch(torch.from_numpy(img0_seq_t).cuda())
    
    kp0_seq, _ = extract_keypoints_nms(scores0_seq, args.nms_radius, args.kp_threshold)
    desc0_seq = sample_descriptors_torch(
        torch.from_numpy(kp0_seq)[None].cuda(), desc0_seq
    )
    
    kps_all.append(kp0_seq)
    descs_all.append(desc0_seq)
    
    # Process remaining images
    for idx, img_path in enumerate(image_names[1:], 1):
        img_seq, _ = load_image(img_path)
        img_seq_t = prepare_image_for_superpoint(img_seq, args.img_h, args.img_w)
        
        with torch.no_grad():
            scores_seq, desc_seq = sp_torch(torch.from_numpy(img_seq_t).cuda())
        
        kp_seq, _ = extract_keypoints_nms(scores_seq, args.nms_radius, args.kp_threshold)
        desc_seq = sample_descriptors_torch(
            torch.from_numpy(kp_seq)[None].cuda(), desc_seq
        )
        
        # Pad to same size for LightGlue
        kp0_lg_seq = torch.from_numpy(kp0_seq)[None].cuda()
        kp1_lg_seq = torch.from_numpy(kp_seq)[None].cuda()
        desc0_lg_seq = desc0_seq
        desc1_lg_seq = desc_seq
        
        if len(kp0_seq) != len(kp_seq):
            min_kpts = min(len(kp0_seq), len(kp_seq))
            kp0_lg_seq = kp0_lg_seq[:, :min_kpts, :]
            kp1_lg_seq = kp1_lg_seq[:, :min_kpts, :]
            desc0_lg_seq = desc0_lg_seq[:, :min_kpts, :]
            desc1_lg_seq = desc1_lg_seq[:, :min_kpts, :]
        
        with torch.amp.autocast('cuda'):
            scores_lg = lg_torch(kp0_lg_seq, kp1_lg_seq, desc0_lg_seq, desc1_lg_seq)
        
        # Extract matches from scores matrix
        m0, m1, mscores0, mscores1 = filter_matches_torch(scores_lg, th=0.1)
        num_matches = (m0 >= 0).sum().item()
        
        matches_all.append({
            'idx': idx,
            'num_kp0': kp0_lg_seq.shape[1],
            'num_kp1': kp1_lg_seq.shape[1],
            'num_matches': num_matches
        })
        
        kps_all.append(kp_seq)
        descs_all.append(desc_seq)
        
        print(f"  Frame {idx}: kp0={kp0_lg_seq.shape[1]}, kp1={kp1_lg_seq.shape[1]}, matches={num_matches}")
    
    # ============== BENCHMARK ==============
    print("\n" + "="*60)
    print("BENCHMARK (SuperPoint FP32)")
    print("="*60)
    
    # Re-prepare TRT inputs for benchmark (fresh allocation)
    sp_inputs_bench = {'image': img0_t.astype(np.float32)}
    sp_input_shapes_bench = {'image': img0_t.shape}
    sp_d_inputs_bench, sp_bindings_bench = allocate_trt_buffers(
        sp_engine, sp_context, sp_input_shapes_bench, {}
    )
    
    # Warmup
    for _ in range(args.warmup):
        with torch.no_grad():
            _ = sp_torch(torch.from_numpy(img0_t).cuda())
        set_trt_inputs(sp_context, sp_d_inputs_bench, sp_inputs_bench)
        sp_context.execute_v2(sp_bindings_bench)
    
    # Custom torch benchmark
    torch_times = []
    for _ in range(args.iter):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = sp_torch(torch.from_numpy(img0_t).cuda())
        torch.cuda.synchronize()
        torch_times.append((time.perf_counter() - start) * 1000)
    
    # TRT benchmark
    trt_times = []
    for _ in range(args.iter):
        start = time.perf_counter()
        set_trt_inputs(sp_context, sp_d_inputs_bench, sp_inputs_bench)
        sp_context.execute_v2(sp_bindings_bench)
        trt_times.append((time.perf_counter() - start) * 1000)
    
    print(f"\nSuperPoint ({args.iter} iterations):")
    print(f"  Custom Torch: {np.mean(torch_times):.2f} ms (std: {np.std(torch_times):.2f})")
    print(f"  Official TRT: {np.mean(trt_times):.2f} ms (std: {np.std(trt_times):.2f})")
    if np.mean(trt_times) > 0:
        print(f"  Speedup: {np.mean(torch_times)/np.mean(trt_times):.2f}x")
    
    print("\n" + "="*60)
    print("Comparison Complete!")
    print("="*60)


def filter_matches_torch(scores: torch.Tensor, th: float):
    """Obtain matches from log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1


if __name__ == '__main__':
    main()
