# main training code
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import matplotlib.pyplot as plt
from Utils import Config
from Utils import utils
from Utils.utils import DataGenerator
from Utils.utils import compute_signed_distance_map, boundary_weighted_bce_loss, gradient_consistency_loss


np.random.seed(12345)
torch.manual_seed(12345)

# GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"{torch.cuda.device_count()} GPU(s) are available for training.")
else:
    device = torch.device('cpu')
    print("No GPU available, using CPU instead.")


parser = argparse.ArgumentParser(description="Choose the model and loss")
parser.add_argument(
    '--model',
    type=str,
    default='UPPA3',
    choices=['UNet', 'UnetPlusPlus', 'UCTransNet', 'CWnet', 'CSWnet', 'UPPA', 'UPPA1', 'UPPA2', 'UPPA3'],
    help='Choose the model (default: UPPA3)'
)
parser.add_argument(
    '--loss',
    type=str,
    default='newloss',
    choices=['BCE', 'Focal', 'WeightedBCE', 'FocalTversky','AdaptiveTverskyFocalLoss', 'WeightedDiceLoss','DWBLossBinary', 'newloss'],
    help='Choose the loss (default: newloss)'
)
parser.add_argument(
    '--data-root',
    type=str,
    default='./Data',
    help='Data root containing GroundTruth and Label directories'
)
parser.add_argument(
    '--output-dir',
    type=str,
    default='./runs',
    help='Directory used to save logs, checkpoints and loss curves'
)
parser.add_argument(
    '--resume',
    type=str,
    default=None,
    help='Checkpoint path used to resume training'
)
parser.add_argument(
    '--debug',
    action='store_true',
    help='Use a small dataset split and train for one epoch'
)
args = parser.parse_args()


# Logger and checkpoint directories
run_name = f'{args.model}_{args.loss}'
if args.debug:
    run_name = f'debug_{run_name}'

run_dir = os.path.join(args.output_dir, run_name)
log_dir = os.path.join(run_dir, 'log')
check_dir = os.path.join(run_dir, 'check')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(check_dir, exist_ok=True)


# Training parameters
sxpath = os.path.join(args.data_root, 'GroundTruth') + os.sep
kxpath = os.path.join(args.data_root, 'Label') + os.sep
n1, n2, n3 = Config.img_size, Config.img_size, Config.img_size

if args.debug:
    tdata_ids = range(1, 3)
    vdata_ids = range(3, 4)
    epochs = 1
else:
    tdata_ids = range(1, 161)
    vdata_ids = range(161, 181)
    epochs = Config.epochs

params = {'dim': (n1, n2, n3), 'n_channels': Config.n_channels, 'shuffle': True}

train_dataset = DataGenerator(
    sxpath,
    kxpath,
    tdata_ids,
    **params,
    Enhance=True,
    SmallScale=False
)
val_dataset = DataGenerator(
    sxpath,
    kxpath,
    vdata_ids,
    **params,
    Enhance=False,
    SmallScale=False
)
train_loader = DataLoader(train_dataset, Config.batch_size, True)
val_loader = DataLoader(val_dataset, Config.batch_size, False)


config = Config.get_CTranS_config()
lr = config.learning_rate


# Loss function
if args.loss == 'BCE':
    criterion = nn.BCELoss()
elif args.loss == 'WeightedBCE':
    criterion = utils.WeightedBCE()
elif args.loss == 'Focal':
    criterion = utils.FocalLoss()
elif args.loss == 'FocalTversky':
    criterion = utils.FocalTverskyLoss()
elif args.loss == 'AdaptiveTverskyFocalLoss':
    criterion = utils.AdaptiveTverskyFocalLoss(
        tversky_weight=0.2,
        focal_weight=0.8,
        total_epoch=epochs
    )
elif args.loss == 'WeightedDiceLoss':
    criterion = utils.WeightedDiceLoss()
elif args.loss == 'DWBLossBinary':
    criterion = utils.DWBLossBinary()
    criterion.w_pos = criterion.w_pos.to(device)
    criterion.w_neg = criterion.w_neg.to(device)
else:
    # newloss uses Focal Loss as the segmentation loss.
    criterion = utils.FocalLoss()


# Load model
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


optimizer = optim.Adam(model.parameters(), lr=lr)
lr_scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=1,
    eta_min=1e-8
)


resume_checkpoint = args.resume
start_epoch = 0
best_val_loss = float('inf')

if resume_checkpoint is not None:
    checkpoint = torch.load(resume_checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))
    else:
        model.load_state_dict(checkpoint)
    print(f"Checkpoint loaded from {resume_checkpoint}")


def calculate_loss(outputs, targets, epoch):
    if args.loss == 'AdaptiveTverskyFocalLoss':
        return criterion(outputs, targets, epoch)

    if args.loss == 'newloss':
        seg_loss = criterion(outputs, targets)

        masks_np = targets.detach().cpu().numpy()[:, 0]
        dist_maps_np = np.stack([
            compute_signed_distance_map(mask_np)
            for mask_np in masks_np
        ])
        dist_map = torch.from_numpy(dist_maps_np).to(device)

        # Boundary weighted BCE
        loss_bnd = boundary_weighted_bce_loss(outputs, targets, dist_map)
        # 3D gradient consistency
        loss_grad = gradient_consistency_loss(outputs, targets)

        return seg_loss + loss_bnd + loss_grad

    return criterion(outputs, targets)


print(f'Model: {args.model}, Loss: {args.loss}, Debug: {args.debug}')
print(f'Training data: {sxpath}')
print(f'Label data: {kxpath}')
print(f'Output directory: {run_dir}')


train_loss_file = os.path.join(log_dir, f'training_loss_{args.model}_{args.loss}.txt')
val_loss_file = os.path.join(log_dir, f'validation_loss_{args.model}_{args.loss}.txt')

if resume_checkpoint is None:
    open(train_loss_file, 'w').close()
    open(val_loss_file, 'w').close()

epochs_list = []
train_losses = []
val_losses = []


for epoch in range(start_epoch, epochs):
    model.train()
    train_loss = 0.0

    for i, (inputs, targets) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Training]", leave=False)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = calculate_loss(outputs, targets, epoch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    with open(train_loss_file, 'a') as f:
        f.write(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}\n")
    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(
                tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Validation]", leave=False)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = calculate_loss(outputs, targets, epoch)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    with open(val_loss_file, 'a') as f:
        f.write(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss}\n")
    print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss}")

    epochs_list.append(epoch + 1)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    lr_scheduler.step(epoch + 1)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_checkpoint_path = os.path.join(
            check_dir,
            f'checkpoint_best_{args.model}_{args.loss}.pth'
        )
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss
        }, best_checkpoint_path)
        print(f"Best checkpoint saved to {best_checkpoint_path}")

    if (epoch + 1) % 25 == 0:
        checkpoint_path = os.path.join(
            check_dir,
            f'checkpoint_epoch_{args.model}_{epoch + 1}_{args.loss}.pth'
        )
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")


final_checkpoint_path = os.path.join(
    check_dir,
    f'checkpoint_final_{args.model}_{args.loss}.pth'
)
torch.save(model.state_dict(), final_checkpoint_path)
print(f"Final model saved to {final_checkpoint_path}")


# Plot loss curves from the current training process
plt.figure(figsize=(10, 6))
plt.plot(epochs_list, train_losses, label='Training Loss', marker='o')
plt.plot(epochs_list, val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Training and Validation Loss ({args.model}, {args.loss})')
plt.legend()
plt.grid(True)
plt.tight_layout()

loss_plot_path = os.path.join(
    run_dir,
    f'loss_convergence_plot_{args.model}_{args.loss}.png'
)
plt.savefig(loss_plot_path, dpi=300)
plt.close()
print(f"Loss curve saved to {loss_plot_path}")
