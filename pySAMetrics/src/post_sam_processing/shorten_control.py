import xarray as xr
import os

base_path = "/Volumes/LaCie/000_POSTDOC_2025"
control_path = f"{base_path}/control"
squall_path = f"{base_path}/squall_line"
output_path = f"{base_path}/control_short"

os.makedirs(output_path, exist_ok=True)

#for dim in ["1d", "2d", "3d"]:
for dim in ["3d"]:
    # Load squall_line dataset for time reference
    squall_ds = xr.open_dataset(f"{squall_path}/dataset_{dim}.nc")
    time_ref = squall_ds["time"]

    # Load control dataset
    control_ds = xr.open_dataset(f"{control_path}/dataset_{dim}.nc")

    # Ensure time is unique
    if not control_ds.indexes['time'].is_unique:
        control_ds = control_ds.sel(time=~control_ds.get_index("time").duplicated())

    # Now select time
    trimmed = control_ds.sel(time=time_ref)

    # Save
    trimmed.to_netcdf(f"{output_path}/dataset_{dim}.nc")
    print(f"Saved: dataset_{dim}.nc")
