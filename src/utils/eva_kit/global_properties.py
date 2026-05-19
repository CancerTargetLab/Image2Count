import dask
import dask.array as da
from dask import delayed
import numpy as np
from src.utils.image_preprocess import load_img

@delayed
def get_properties(img_path, channel_masking, mpp):
    img_data = load_img(img_path,'').astype(np.float32)
    min_vals = np.min(img_data, axis=(1, 2), keepdims=True)
    max_vals = np.max(img_data, axis=(1, 2), keepdims=True)
    return min_vals, max_vals

def get_global_properties(img_paths):

    tasks = [get_properties(p, channel_masking, mpp) for p in img_paths]
    results = dask.compute(*tasks)
    mins = [mn for (mn, mx) in results]
    maxs = [mx for (mn, mx) in results]

    min_vals = np.min(np.stack(mins, axis=0), axis=0)
    max_vals = np.max(np.stack(maxs, axis=0), axis=0)   

    return min_vals, ptp_vals, img_max
