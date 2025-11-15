import os


import xarray as xr
for i in range(2, 48):
    try:
        ds = xr.open_dataset(f'/Volumes/LaCie/000_POSTDOC_2025/long_high_res/3D_test/split_{i}.nc')
        print(f"✅ Success {i}")
    except:
        print(f"fail {i}")

