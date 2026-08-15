import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.signal import fftconvolve  # Using FFT to accelerate convolution


# -------------------------------
# generate synthetic_seismic data
# -------------------------------
def generate_synthetic_seismic_final(
        nx,
        ny,
        nt,
        num_layers,
        num_faults,
        num_anomalies,
        dip_angle=20,
        intermediate_dir=None,
        save_intermediate_dat=True
):
    intermediate_dir = Path(intermediate_dir) if intermediate_dir is not None else None

    def save_intermediate(array, filename):
        if not save_intermediate_dat:
            return
        if intermediate_dir is None:
            raise ValueError("intermediate_dir must be provided when save_intermediate_dat is enabled")
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        array.astype('float32').tofile(intermediate_dir / filename)

    # Step1
    z = np.arange(nt).reshape(1, 1, -1)
    data3D = np.zeros((nx, ny, nt), dtype=np.float32)
    label = np.zeros((nx, ny, nt), dtype=np.float32)
    layer_values = np.random.uniform(0, 0.01, num_layers)
    save_intermediate(data3D, "Step1.dat")

    # Step2
    global_interfaces = np.linspace(0, nt, num_layers + 1)
    interfaces = [global_interfaces[0] * np.ones((nx, ny))]

    for i in range(1, num_layers):
        global_inc = global_interfaces[i] - global_interfaces[i - 1]
        inc_smooth = gaussian_filter(np.random.rand(nx, ny), sigma=20)
        inc_smooth = (inc_smooth - inc_smooth.min()) / np.ptp(inc_smooth) - 0.5   # [0.5,-0.5]
        interfaces.append(interfaces[-1] + global_inc + 7 * inc_smooth)
    interfaces.append(global_interfaces[-1] * np.ones((nx, ny)))

    fault_params_list = []
    for _ in range(num_faults):
        max_delta_z = np.random.uniform(20, 40)
        sigma = np.random.uniform(0.5, 2.0)
        fault_start_z0 = (np.random.randint(0, nx // 2), np.random.randint(0, ny // 2))
        fault_end_z0 = (np.random.randint(nx // 2, nx), np.random.randint(ny // 2, ny))
        fault_start_z1 = (fault_start_z0[0] - np.random.randint(10, 30), fault_start_z0[1])
        fault_end_z1 = (fault_end_z0[0] - np.random.randint(10, 30), fault_end_z0[1])
        for i in range(1, num_layers):
            t = i / num_layers

            current_start = (
                int(fault_start_z0[0] * (1 - t) + fault_start_z1[0] * t),
                int(fault_start_z0[1] * (1 - t) + fault_start_z1[1] * t)
            )
            current_end = (
                int(fault_end_z0[0] * (1 - t) + fault_end_z1[0] * t),
                int(fault_end_z0[1] * (1 - t) + fault_end_z1[1] * t)
            )
            ax, ay = current_end[0] - current_start[0], current_end[1] - current_start[1] # 计算断层方向向量
            interface = interfaces[i].copy()

            ix, iy = np.arange(nx), np.arange(ny)
            ix_grid, iy_grid = np.meshgrid(ix, iy, indexing='ij')
            px = ix_grid - current_start[0]
            py = iy_grid - current_start[1]

            denominator = ax ** 2 + ay ** 2
            t_local = (px * ax + py * ay) / denominator
            t_clamped = np.clip(t_local, 0.0, 1.0)
            cross = ax * py - ay * px
            delta_z = max_delta_z * np.sin(t_clamped * np.pi) * (1 - t)
            delta_z_field = np.where(cross > 0, delta_z, 0)
            delta_z_field = gaussian_filter(delta_z_field, sigma=sigma)

            interfaces[i] = interface + delta_z_field

        fault_params_list.append({
            "fault_start_z0": fault_start_z0,
            "fault_end_z0": fault_end_z0,
            "fault_start_z1": fault_start_z1,
            "fault_end_z1": fault_end_z1,
            "max_delta_z": max_delta_z,
            "sigma": sigma
        })

    interfaces_arr = np.clip(np.sort(np.stack(interfaces, axis=0).astype(int), axis=0), 0, nt)

    for i in range(num_layers):
        lower = interfaces_arr[i].reshape(nx, ny, 1)
        upper = interfaces_arr[i + 1].reshape(nx, ny, 1)
        mask = (z >= lower) & (z < upper)
        data3D[mask] = layer_values[i]

    save_intermediate(data3D, "Step2.dat")

    # Step3
    x = np.linspace(0, 2 * np.pi, nx)
    y = np.linspace(0, 2 * np.pi, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    for i in range(num_layers):
        phase = np.random.uniform(0, np.pi)
        sin_wave = np.sin(2 * xx + phase) + np.cos(2 * yy + phase)
        sin_wave = sin_wave / np.ptp(sin_wave) * 0.005
        gauss_wave = np.exp(-((xx - np.pi) ** 2 + (yy - np.pi) ** 2) / (2 * 10 ** 2)) * 0.005
        variation = (sin_wave + gauss_wave).astype(np.float32)

        lower = interfaces_arr[i]
        upper = interfaces_arr[i + 1]
        z_mask = (z >= lower[..., np.newaxis]) & (z < upper[..., np.newaxis])
        data3D += variation[..., np.newaxis] * z_mask

    save_intermediate(data3D, "Step3.dat")

    # Step4
    tan_alpha = np.tan(np.radians(dip_angle))
    x0 = fault_params_list[0]["fault_start_z0"]
    anomalies = []
    for _ in range(num_anomalies):
        base_z = np.random.uniform(0, nt)
        cx = int(x0[0] + base_z * tan_alpha + np.random.uniform(-20, 20))
        cy = int(x0[1] + np.random.uniform(-20, 20))
        cz = np.random.randint(30, 226)
        sx, sy = np.random.randint(4, 6, 2)
        sz = int(sx / np.sin(np.radians(dip_angle)))
        anomalies.append((cx, cy, cz, sx, sy, sz))

        xx, yy, zz = np.ogrid[:nx, :ny, :nt]
        x_rot = (xx - cx) * np.cos(np.radians(dip_angle)) + (zz - cz) * np.sin(np.radians(dip_angle))
        z_rot = -(xx - cx) * np.sin(np.radians(dip_angle)) + (zz - cz) * np.cos(np.radians(dip_angle))
        dist = (x_rot / sx) ** 2 + ((yy - cy) / sy) ** 2 + (z_rot / sz) ** 2
        anomaly_mask = dist < 1
        data3D[anomaly_mask] -= 0.004
        label[anomaly_mask] = 1

    save_intermediate(data3D, "Step3.dat")

    # Step5
    data3D += np.random.normal(0, 0.0002, data3D.shape)
    save_intermediate(data3D, "Step4.dat")

    # Step6
    relative_impedance = (data3D - data3D.min()) / np.ptp(data3D) * 0.01
    epsilon = 1e-6
    R = np.diff(relative_impedance, axis=2) / (relative_impedance[..., :-1] + relative_impedance[..., 1:] + epsilon)
    R = np.pad(R, ((0, 0), (0, 0), (0, 1)), mode='edge')
    save_intermediate(R, "Step5.dat")
    t_wave = np.linspace(-0.055, 0.055, 56)
    rickwavelet = (1 - 2 * (np.pi * 25 * t_wave) ** 2) * np.exp(-(np.pi * 25 * t_wave) ** 2)
    rickwavelet = rickwavelet / np.abs(rickwavelet).max()
    synthetic_seismic = fftconvolve(R, rickwavelet[np.newaxis, np.newaxis, :], mode='same', axes=2)
    save_intermediate(synthetic_seismic, "Step6.dat")
    # step 7
    synthetic_seismic_final = gaussian_filter(synthetic_seismic, sigma=0.6)
    save_intermediate(synthetic_seismic_final, "Step7.dat")
    return synthetic_seismic_final, label, anomalies, fault_params_list


def parse_args():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate synthetic 3D seismic volumes and labels.")
    parser.add_argument("--nx", type=int, default=256, help="Number of voxels along the x-axis.")
    parser.add_argument("--ny", type=int, default=256, help="Number of voxels along the y-axis.")
    parser.add_argument("--nt", type=int, default=256, help="Number of voxels along the time/depth axis.")
    parser.add_argument("--num-layers", type=int, default=80, help="Number of geological layers.")
    parser.add_argument("--num-faults", type=int, default=2, help="Number of simulated faults.")
    parser.add_argument("--num-anomalies", type=int, default=60, help="Number of simulated anomalies per volume.")
    parser.add_argument("--num-data", type=int, default=1, help="Number of data-label pairs to generate.")
    parser.add_argument("--dip-angle", type=float, default=20, help="Dip angle of the simulated anomalies in degrees.")
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=project_root / "Data" / "GroundTruth",
        help="Directory used to store generated seismic volumes."
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=project_root / "Data" / "Label",
        help="Directory used to store generated segmentation labels."
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=project_root / "Picture" / "ModelingProcess",
        help="Directory used to store intermediate modeling-process DAT files."
    )
    parser.add_argument(
        "--save-intermediate-dat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save intermediate Step1.dat to Step7.dat files."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.ground_truth_dir.mkdir(parents=True, exist_ok=True)
    args.label_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for i in range(args.num_data):
        synthetic_seismic_final, label, anomalies, fault_params_list = generate_synthetic_seismic_final(
            args.nx,
            args.ny,
            args.nt,
            args.num_layers,
            args.num_faults,
            args.num_anomalies,
            dip_angle=args.dip_angle,
            intermediate_dir=args.intermediate_dir,
            save_intermediate_dat=args.save_intermediate_dat
        )
        synthetic_seismic_final.astype('float32').tofile(
            args.ground_truth_dir / f"synthetic_seismic_final_{i + 1}.dat"
        )
        label.astype('float32').tofile(
            args.label_dir / f"synthetic_seismic_final_{i + 1}.dat"
        )
        print(f" {i + 1} data saved")


if __name__ == "__main__":
    main()
