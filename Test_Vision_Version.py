import os
import argparse
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from Utils import Config
from Utils.utils import DataGenerator


# GPU Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"{torch.cuda.device_count()} GPU(s) are available for visualization.")
else:
    print("No GPU available, using CPU instead.")


model_choices = [
    'GSCDUnet', 'UNet', 'UnetPlusPlus', 'UCTransNet', 'CWnet', 'CSWnet',
    'UPPA', 'UPPA1', 'UPPA2', 'UPPA3'
]

parser = argparse.ArgumentParser(description="Visualize synthetic-data segmentation results")
parser.add_argument(
    '--model',
    type=str,
    default='GSCDUnet',
    choices=model_choices,
    help='Choose the model; UPPA3 is the legacy name of GSCDUnet'
)
parser.add_argument(
    '--checkpoint',
    type=str,
    required=True,
    help='Path to a model state_dict or training checkpoint'
)
parser.add_argument(
    '--data-root',
    type=str,
    default='./Data',
    help='Data root containing GroundTruth and Label directories'
)
parser.add_argument(
    '--test-data-ids',
    type=int,
    nargs='+',
    default=[181],
    help='Data IDs to visualize, for example: --test-data-ids 181 182'
)
parser.add_argument(
    '--dim',
    type=int,
    nargs=3,
    default=[256, 256, 256],
    metavar=('N1', 'N2', 'N3'),
    help='Shape of each source volume (default: 256 256 256)'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=0.5,
    help='Probability threshold used to generate the binary prediction'
)
args = parser.parse_args()


if not os.path.isfile(args.checkpoint):
    parser.error(f'checkpoint does not exist: {args.checkpoint}')

if not 0.0 <= args.threshold <= 1.0:
    parser.error('--threshold must be between 0 and 1')

if any(size <= 0 for size in args.dim):
    parser.error('--dim values must be positive')


def build_model(model_name):
    model_module = importlib.import_module(f'Models.{model_name}')
    model_class = getattr(model_module, model_name)

    if model_name in ['UCTransNet', 'CWnet', 'CSWnet']:
        model = model_class(Config.get_CTranS_config())
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


model = build_model(args.model)
load_checkpoint(model, args.checkpoint)
print(f"Model loaded from {args.checkpoint}")


# Testing parameters
sxpath = os.path.join(args.data_root, 'GroundTruth') + os.sep
kxpath = os.path.join(args.data_root, 'Label') + os.sep
n1, n2, n3 = args.dim
test_data_ids = args.test_data_ids
params = {'dim': (n1, n2, n3), 'n_channels': Config.n_channels, 'shuffle': False}

test_dataset = DataGenerator(
    dpath=sxpath,
    fpath=kxpath,
    data_IDs=test_data_ids,
    **params,
    Enhance=False,
    SmallScale=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)


import cigvis

nodes = []
sample_count = 0

with torch.no_grad():
    for idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        pred_mask = (outputs > args.threshold).float()

        inputs_np = inputs.cpu().numpy()[0, 0].astype(np.float32)
        targets_np = targets.cpu().numpy()[0, 0].astype(np.float32)
        outputs_np = outputs.cpu().numpy()[0, 0].astype(np.float32)
        pred_mask_np = pred_mask.cpu().numpy()[0, 0].astype(np.float32)

        data_id = test_data_ids[idx]
        print(f"Sample ID {data_id}:")
        print(f"Input min: {inputs_np.min()}, max: {inputs_np.max()}")
        print(f"Target min: {targets_np.min()}, max: {targets_np.max()}")
        print(f"Output min: {outputs_np.min()}, max: {outputs_np.max()}")
        print(f"Binary min: {pred_mask_np.min()}, max: {pred_mask_np.max()}")

        nodes.extend([
            cigvis.create_slices(inputs_np, cmap='Petrel', name=f'ID {data_id} - Input'),
            cigvis.create_slices(targets_np, cmap='Petrel', name=f'ID {data_id} - Label'),
            cigvis.create_slices(outputs_np, cmap='Petrel', name=f'ID {data_id} - Probability'),
            cigvis.create_slices(pred_mask_np, cmap='Petrel', name=f'ID {data_id} - Prediction')
        ])
        sample_count += 1


if sample_count == 0:
    raise RuntimeError('No samples were loaded for visualization')

cigvis.plot3D(nodes, grid=[sample_count, 4], share=True)
