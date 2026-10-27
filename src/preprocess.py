import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import yaml

def load_config(config_path='configs/default.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def resample_3d(image_array, target_shape=(96, 96, 96)):
    tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=target_shape, mode='trilinear', align_corners=True)
    return resized.squeeze(0).squeeze(0).numpy()

def process_mri(img_path, target_res=96):
    img = nib.load(img_path)
    data = img.get_fdata()
    if len(data.shape) == 4 and data.shape[-1] == 1:
        data = np.squeeze(data, axis=-1)
    resampled = resample_3d(data, target_shape=(target_res, target_res, target_res))
    mask = resampled > 0
    if mask.sum() > 0:
        mean_val = resampled[mask].mean()
        std_val = resampled[mask].std() + 1e-8
        normalized = np.zeros_like(resampled)
        normalized[mask] = (resampled[mask] - mean_val) / std_val
    else:
        normalized = resampled
    return normalized.astype(np.float16)

def preprocess_all():
    config = load_config()
    raw_dir = config['paths']['raw_dir']
    processed_dir = config['paths']['processed_dir']
    target_res = config['preprocessing']['target_resolution']
    os.makedirs(processed_dir, exist_ok=True)
    pattern = os.path.join(raw_dir, "**/PROCESSED/MPRAGE/T88_111/*_masked_gfc.img")
    mri_paths = glob.glob(pattern, recursive=True)
    print(f"Found {len(mri_paths)} raw MRI files matching masked GFC pattern.")
    count = 0
    for p in mri_paths:
        subject_id = os.path.basename(p).split('_mpr')[0]
        out_name = f"{subject_id}.npy"
        out_path = os.path.join(processed_dir, out_name)
        if os.path.exists(out_path):
            continue
        try:
            processed_volume = process_mri(p, target_res=target_res)
            np.save(out_path, processed_volume)
            count += 1
        except Exception as e:
            print(f"Error processing file {p}: {e}")
    print(f"Preprocessing complete. Processed {count} new files. Total files in target dir: {len(os.listdir(processed_dir))}")
