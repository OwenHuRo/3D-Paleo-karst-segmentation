import argparse
from pathlib import Path

import numpy as np


def parse_arguments():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Inspect generated seismic volumes and paleokarst labels."
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=181,
        help="First data index to load (inclusive).",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=182,
        help="Last data index to load (exclusive).",
    )
    parser.add_argument(
        "--data-prefix",
        type=str,
        default=str(project_root / "Data" / "GroundTruth" / "synthetic_seismic_final"),
        help="File prefix for seismic data.",
    )
    parser.add_argument(
        "--label-prefix",
        type=str,
        default=str(project_root / "Data" / "Label" / "synthetic_seismic_final"),
        help="File prefix for label data.",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        default=(256, 256, 256),
        metavar=("N1", "N2", "N3"),
        help="Volume dimensions.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="NumPy data type used to read binary files.",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="Petrel",
        help="Colormap used for visualization.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        nargs=2,
        default=None,
        metavar=("ROWS", "COLUMNS"),
        help="Visualization grid. By default, each sample occupies one row.",
    )
    parser.add_argument(
        "--share",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Share camera and interaction settings between subplots.",
    )
    parser.add_argument(
        "--print-class-percentages",
        action="store_true",
        help="Print background and target percentages for each label volume.",
    )

    return parser.parse_args()


def build_file_path(file_prefix, data_id):
    return Path(f"{file_prefix}_{data_id}.dat")


def load_volume(file_path, volume_shape, data_type):
    if not file_path.is_file():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    volume = np.fromfile(file_path, dtype=data_type)
    expected_size = int(np.prod(volume_shape))

    if volume.size != expected_size:
        raise ValueError(
            f"Invalid data size in {file_path}: expected {expected_size} values, "
            f"but found {volume.size}."
        )

    return volume.reshape(volume_shape)


def load_data_pair(
    data_id,
    data_prefix,
    label_prefix,
    volume_shape,
    data_type,
):
    data_path = build_file_path(data_prefix, data_id)
    label_path = build_file_path(label_prefix, data_id)

    seismic_volume = load_volume(data_path, volume_shape, data_type)
    label_volume = load_volume(label_path, volume_shape, data_type)

    return seismic_volume, label_volume


def print_class_percentages(data_id, label_volume):
    target_count = np.count_nonzero(label_volume)
    total_count = label_volume.size
    background_count = total_count - target_count

    target_percentage = target_count / total_count * 100
    background_percentage = background_count / total_count * 100

    print(
        f"Data {data_id}: background = {background_percentage:.2f}%, "
        f"target = {target_percentage:.2f}%"
    )


def create_visualization_nodes(volumes, colormap):
    import cigvis

    return [
        cigvis.create_slices(volume, cmap=colormap)
        for volume in volumes
    ]


def main():
    args = parse_arguments()
    import cigvis

    if args.end_index <= args.start_index:
        raise ValueError(
            "--end-index must be greater than --start-index."
        )

    volume_shape = tuple(args.shape)
    data_type = np.dtype(args.dtype)
    volumes = []

    for data_id in range(args.start_index, args.end_index):
        seismic_volume, label_volume = load_data_pair(
            data_id=data_id,
            data_prefix=args.data_prefix,
            label_prefix=args.label_prefix,
            volume_shape=volume_shape,
            data_type=data_type,
        )

        if args.print_class_percentages:
            print_class_percentages(data_id, label_volume)

        volumes.extend((seismic_volume, label_volume))

    nodes = create_visualization_nodes(volumes, args.colormap)

    # Use one row per sample unless a custom grid is provided.
    sample_count = args.end_index - args.start_index
    grid = list(args.grid) if args.grid else [sample_count, 2]

    if grid[0] * grid[1] < len(nodes):
        raise ValueError(
            f"The grid {grid} cannot contain all {len(nodes)} visualization nodes."
        )

    cigvis.plot3D(nodes, grid=grid, share=args.share)


if __name__ == "__main__":
    main()
