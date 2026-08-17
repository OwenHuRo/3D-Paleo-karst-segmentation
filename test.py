import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Utils import Config
from Utils.utils import DataGenerator


# GPU Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"{torch.cuda.device_count()} GPU(s) are available for testing.")
else:
    print("No GPU available, using CPU instead.")


parser = argparse.ArgumentParser(description="Choose the model and checkpoint for testing")
parser.add_argument(
    '--model',
    type=str,
    default='UPPA3',
    choices=['UNet', 'UnetPlusPlus', 'UCTransNet', 'CWnet', 'CSWnet', 'UPPA', 'UPPA1', 'UPPA2', 'UPPA3'],
    help='Choose the model (default: UPPA3)'
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
    default=[1, 2],
    help='Test data IDs, for example: --test-data-ids 181 182 183'
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
parser.add_argument(
    '--output-dir',
    type=str,
    default='./test_results',
    help='Directory used to save metrics and the ROC curve'
)
args = parser.parse_args()


if not 0.0 <= args.threshold <= 1.0:
    parser.error('--threshold must be between 0 and 1')

if not os.path.isfile(args.checkpoint):
    parser.error(f'checkpoint does not exist: {args.checkpoint}')

os.makedirs(args.output_dir, exist_ok=True)


# Load model
config = Config.get_CTranS_config()

if args.model == 'UNet':
    from Models import UNet
    model = UNet.UNet(input_channels=1, output_channels=1).to(device)
elif args.model == 'UnetPlusPlus':
    from Models import UnetPlusPlus
    model = UnetPlusPlus.UnetPlusPlus(input_channels=1, output_channels=1).to(device)
elif args.model == 'UCTransNet':
    from Models import UCTransNet
    model = UCTransNet.UCTransNet(config).to(device)
elif args.model == 'CWnet':
    from Models import CWnet
    model = CWnet.CWnet(config).to(device)
elif args.model == 'CSWnet':
    from Models import CSWnet
    model = CSWnet.CSWnet(config).to(device)
elif args.model == 'UPPA':
    from Models import UPPA
    model = UPPA.UPPA(input_channels=1, output_channels=1).to(device)
elif args.model == 'UPPA1':
    from Models import UPPA1
    model = UPPA1.UPPA1(input_channels=1, output_channels=1).to(device)
elif args.model == 'UPPA2':
    from Models import UPPA2
    model = UPPA2.UPPA2(input_channels=1, output_channels=1).to(device)
else:
    from Models import UPPA3
    model = UPPA3.UPPA3(input_channels=1, output_channels=1).to(device)


checkpoint = torch.load(args.checkpoint, map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

# Support checkpoints saved from DataParallel.
if any(key.startswith('module.') for key in state_dict):
    state_dict = {
        key.removeprefix('module.'): value
        for key, value in state_dict.items()
    }

model.load_state_dict(state_dict)
model.eval()
print(f"Model loaded from {args.checkpoint}")


# Testing parameters
sxpath = os.path.join(args.data_root, 'GroundTruth') + os.sep
kxpath = os.path.join(args.data_root, 'Label') + os.sep
n1, n2, n3 = args.dim
test_data_ids = args.test_data_ids
params = {'dim': (n1, n2, n3), 'n_channels': Config.n_channels, 'shuffle': False}

# Data Generator for test data. Enhance=False keeps the test crop deterministic.
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
    batch_size=Config.batch_size,
    shuffle=False
)


# Streaming confusion matrix. This replaces the three large Python lists used
tp = 0
tn = 0
fp = 0
fn = 0

# Streaming histogram for a memory-efficient approximate ROC/AUC.
auc_bins = 4096
auc_edges = np.linspace(0.0, 1.0, auc_bins + 1)
positive_hist = np.zeros(auc_bins, dtype=np.int64)
negative_hist = np.zeros(auc_bins, dtype=np.int64)


with torch.no_grad():
    for idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        pred_mask = outputs > args.threshold
        target_mask = targets > 0.5

        tp += torch.logical_and(pred_mask, target_mask).sum().item()
        tn += torch.logical_and(~pred_mask, ~target_mask).sum().item()
        fp += torch.logical_and(pred_mask, ~target_mask).sum().item()
        fn += torch.logical_and(~pred_mask, target_mask).sum().item()

        # Only one batch is kept in NumPy at a time. The probability values are
        # summarized into fixed-size histograms instead of being stored globally.
        scores_np = outputs.detach().float().clamp(0, 1).cpu().numpy().reshape(-1)
        labels_np = target_mask.cpu().numpy().reshape(-1)
        positive_hist += np.histogram(scores_np[labels_np], bins=auc_edges)[0]
        negative_hist += np.histogram(scores_np[~labels_np], bins=auc_edges)[0]

        print(f"Sample batch {idx + 1}/{len(test_loader)}:")
        print(f"Input min: {inputs.min().item()}, max: {inputs.max().item()}")
        print(f"Target min: {targets.min().item()}, max: {targets.max().item()}")
        print(f"Output min: {outputs.min().item()}, max: {outputs.max().item()}")


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


# Calculate metrics from the accumulated confusion matrix.
total = tp + tn + fp + fn
accuracy = safe_divide(tp + tn, total)
precision = safe_divide(tp, tp + fp)
recall = safe_divide(tp, tp + fn)
f1 = safe_divide(2 * tp, 2 * tp + fp + fn)
iou = safe_divide(tp, tp + fp + fn)
background_iou = safe_divide(tn, tn + fp + fn)
miou = (iou + background_iou) / 2.0
dice_coef = f1


# Build an approximate ROC curve by traversing score bins from high to low.
positive_count = positive_hist.sum()
negative_count = negative_hist.sum()

if positive_count > 0 and negative_count > 0:
    cumulative_tp = np.cumsum(positive_hist[::-1], dtype=np.float64)
    cumulative_fp = np.cumsum(negative_hist[::-1], dtype=np.float64)
    tpr = np.concatenate(([0.0], cumulative_tp / positive_count))
    fpr = np.concatenate(([0.0], cumulative_fp / negative_count))
    roc_auc = np.trapz(tpr, fpr)
else:
    tpr = np.array([0.0, 1.0])
    fpr = np.array([0.0, 1.0])
    roc_auc = float('nan')
    print("AUC cannot be calculated because the test labels contain only one class.")


print("Evaluation Metrics:")
print(f"AUC (approx.): {roc_auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"IoU: {iou:.4f}")
print(f"mIoU: {miou:.4f}")
print(f"Dice Coefficient: {dice_coef:.4f}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")


metrics_path = os.path.join(args.output_dir, f'metrics_{args.model}.txt')
with open(metrics_path, 'w') as f:
    f.write("Evaluation Settings:\n")
    f.write(f"Model: {args.model}\n")
    f.write(f"Checkpoint: {args.checkpoint}\n")
    f.write(f"Data root: {args.data_root}\n")
    f.write(f"Test data IDs: {test_data_ids}\n")
    f.write(f"Volume dim: {(n1, n2, n3)}\n")
    f.write(f"Threshold: {args.threshold}\n")
    f.write("\nEvaluation Metrics:\n")
    f.write(f"AUC (approx.): {roc_auc:.4f}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n")
    f.write(f"IoU: {iou:.4f}\n")
    f.write(f"mIoU: {miou:.4f}\n")
    f.write(f"Dice Coefficient: {dice_coef:.4f}\n")
    f.write(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\n")
print(f"Metrics saved to {metrics_path}")


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', linewidth=2,
         label=f'ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', linewidth=2,
         linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic ({args.model})')
plt.legend()
plt.grid(True)
plt.tight_layout()

roc_path = os.path.join(args.output_dir, f'roc_{args.model}.png')
plt.savefig(roc_path, dpi=300)
plt.close()
print(f"ROC curve saved to {roc_path}")
