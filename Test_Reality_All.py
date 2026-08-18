import os
import argparse
import importlib
import numpy as np
import torch
from tqdm import tqdm

from Utils import Config


np.random.seed(12345)
torch.manual_seed(12345)

# GPU Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"{torch.cuda.device_count()} GPU(s) are available for inference.")
else:
    print("No GPU available, using CPU instead.")


model_choices = [
    'GSCDUnet', 'UNet', 'UnetPlusPlus', 'UCTransNet', 'CWnet', 'CSWnet', 'UPPA', 'UPPA1', 'UPPA2', 'UPPA3'
]

parser = argparse.ArgumentParser(description="Run full-volume inference on real seismic data")
parser.add_argument(
    '--models',
    type=str,
    nargs='+',
    default=['GSCDUnet'],
    choices=model_choices,
    help='Models to evaluate, in the same order as --checkpoints'
)
parser.add_argument(
    '--checkpoints',
    type=str,
    nargs='+',
    required=True,
    help='Checkpoint paths corresponding to --models'
)
parser.add_argument(
    '--data-path',
    type=str,
    required=True,
    help='Path to the real SEG-Y or float32 DAT volume'
)
parser.add_argument(
    '--input-format',
    type=str,
    default='segy',
    choices=['segy', 'dat'],
    help='Input volume format (default: segy)'
)
parser.add_argument(
    '--dim',
    type=int,
    nargs=3,
    default=[582, 608, 1039],
    metavar=('N1', 'N2', 'N3'),
    help='Expected volume shape; required to reshape DAT input'
)
parser.add_argument(
    '--block-size',
    type=int,
    nargs=3,
    default=[128, 128, 128],
    metavar=('B1', 'B2', 'B3'),
    help='Inference block size (default: 128 128 128)'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=0.5,
    help='Probability threshold used to save the binary prediction'
)
parser.add_argument(
    '--normalization',
    type=str,
    default='block',
    choices=['block', 'global', 'none'],
    help='Input normalization: per-block, configured global statistics, or none'
)
parser.add_argument(
    '--mean',
    type=float,
    default=4.4420,
    help='Global mean used when --normalization global'
)
parser.add_argument(
    '--std',
    type=float,
    default=1786.6514,
    help='Global standard deviation used when --normalization global'
)
parser.add_argument(
    '--output-dir',
    type=str,
    default='./reality_results',
    help='Directory used to save full-volume prediction DAT files'
)
args = parser.parse_args()


if len(args.models) != len(args.checkpoints):
    parser.error('--models and --checkpoints must contain the same number of values')

if not os.path.isfile(args.data_path):
    parser.error(f'data file does not exist: {args.data_path}')

for checkpoint_path in args.checkpoints:
    if not os.path.isfile(checkpoint_path):
        parser.error(f'checkpoint does not exist: {checkpoint_path}')

if not 0.0 <= args.threshold <= 1.0:
    parser.error('--threshold must be between 0 and 1')

if any(size <= 0 for size in args.dim):
    parser.error('--dim values must be positive')

if any(size <= 0 for size in args.block_size):
    parser.error('--block-size values must be positive')

if any(size % 16 != 0 for size in args.block_size):
    parser.error('--block-size values must be divisible by 16 for the current model family')

transformer_models = {'UCTransNet', 'CWnet', 'CSWnet'}
if transformer_models.intersection(args.models) and len(set(args.block_size)) != 1:
    parser.error('Transformer models require a cubic --block-size')

if args.normalization == 'global' and args.std <= 0:
    parser.error('--std must be positive when using global normalization')

os.makedirs(args.output_dir, exist_ok=True)


def load_volume(data_path, input_format, expected_dim):
    if input_format == 'segy':
        import segyio
        with segyio.open(data_path, 'r') as segyfile:
            volume = segyio.tools.cube(segyfile)
        volume = np.asarray(volume, dtype=np.float32)
    else:
        expected_count = int(np.prod(expected_dim))
        expected_bytes = expected_count * np.dtype(np.float32).itemsize
        actual_bytes = os.path.getsize(data_path)
        if actual_bytes != expected_bytes:
            raise ValueError(
                f'DAT size mismatch: expected {expected_bytes} bytes, got {actual_bytes}'
            )
        volume = np.memmap(
            data_path,
            dtype=np.float32,
            mode='r',
            shape=tuple(expected_dim)
        )

    if tuple(volume.shape) != tuple(expected_dim):
        raise ValueError(
            f'Volume shape mismatch: expected {tuple(expected_dim)}, got {volume.shape}'
        )
    return volume


def build_model(model_name):
    model_module = importlib.import_module(f'Models.{model_name}')
    model_class = getattr(model_module, model_name)

    if model_name in transformer_models:
        model = model_class(
            Config.get_CTranS_config(),
            img_size=args.block_size[0]
        )
    else:
        model = model_class(input_channels=1, output_channels=1)
    return model.to(device)


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    if any(key.startswith('module.') for key in state_dict):
        state_dict = {
            key.removeprefix('module.'): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.eval()


def normalize_block(block):
    if args.normalization == 'block':
        block_mean = np.mean(block)
        block_std = np.std(block)
        return (block - block_mean) / (block_std + 1e-8)

    if args.normalization == 'global':
        return (block - args.mean) / (args.std + 1e-8)

    return block


def block_starts(length, block_length):
    return range(0, length, block_length)


volume = load_volume(args.data_path, args.input_format, args.dim)
n1, n2, n3 = volume.shape
b1, b2, b3 = args.block_size

x_starts = list(block_starts(n1, b1))
y_starts = list(block_starts(n2, b2))
z_starts = list(block_starts(n3, b3))
total_blocks = len(x_starts) * len(y_starts) * len(z_starts)

print(f'Real volume loaded from {args.data_path}')
print(f'Volume shape: {volume.shape}')
print(f'Block size: {(b1, b2, b3)}, total blocks per model: {total_blocks}')
print(f'Normalization: {args.normalization}')


# Process models one at a time so only one model occupies GPU memory. Each
# prediction is written directly to a disk-backed array instead of allocating
# one full in-memory prediction volume per model.
for model_name, checkpoint_path in zip(args.models, args.checkpoints):
    model = build_model(model_name)
    load_checkpoint(model, checkpoint_path)
    print(f'Model {model_name} loaded from {checkpoint_path}')

    output_path = os.path.join(args.output_dir, f'pred_{model_name}.dat')
    prediction = np.memmap(
        output_path,
        dtype=np.float32,
        mode='w+',
        shape=volume.shape
    )

    progress = tqdm(total=total_blocks, desc=f'{model_name} [Inference]')

    with torch.no_grad():
        for x in x_starts:
            x_end = min(x + b1, n1)
            for y in y_starts:
                y_end = min(y + b2, n2)
                for z in z_starts:
                    z_end = min(z + b3, n3)

                    valid_block = np.asarray(
                        volume[x:x_end, y:y_end, z:z_end],
                        dtype=np.float32
                    )
                    valid_block = normalize_block(valid_block).astype(np.float32)

                    # Pad the final block on each axis to the configured model
                    # input size, then crop the prediction back to valid data.
                    input_block = np.zeros((b1, b2, b3), dtype=np.float32)
                    vx, vy, vz = valid_block.shape
                    input_block[:vx, :vy, :vz] = valid_block

                    inputs = torch.from_numpy(input_block).unsqueeze(0).unsqueeze(0).to(device)
                    outputs = model(inputs)
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]

                    pred = (outputs > args.threshold).float()[0, 0]
                    pred = pred[:vx, :vy, :vz].cpu().numpy()
                    prediction[x:x_end, y:y_end, z:z_end] = pred
                    progress.update(1)

    progress.close()
    prediction.flush()
    output_size_gb = os.path.getsize(output_path) / (1024 ** 3)
    print(f'Saved {output_path} ({output_size_gb:.2f} GB)')

    del prediction
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


print('Full-volume inference completed.')
