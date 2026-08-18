# main training code
import os
import argparse
import random
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


# GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"{torch.cuda.device_count()} GPU(s) are available for training.")
else:
    device = torch.device('cpu')
    print("No GPU available, using CPU instead.")


parser = argparse.ArgumentParser(description="Train 3D paleokarst cave segmentation models")
parser.add_argument(
    '--model',
    type=str,
    default='GSCDUnet',
    choices=['GSCDUnet', 'UNet', 'UnetPlusPlus', 'UCTransNet', 'CWnet', 'CSWnet', 'UPPA', 'UPPA1', 'UPPA2', 'UPPA3'],
    help='Choose the model; UPPA3 is the legacy name of GSCDUnet'
)
parser.add_argument(
    '--loss',
    type=str,
    default='CompositeLoss',
    choices=['CompositeLoss', 'BCE', 'Focal', 'WeightedBCE', 'FocalTversky', 'AdaptiveTverskyFocalLoss', 'WeightedDiceLoss', 'DWBLossBinary', 'newloss'],
    help='Choose the loss; newloss is the legacy name of CompositeLoss'
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
parser.add_argument(
    '--seed',
    type=int,
    default=12345,
    help='Random seed used for training and data augmentation'
)
args = parser.parse_args()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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


def validate_data_files(data_ids):
    missing_files = []
    for data_id in data_ids:
        data_path = os.path.join(sxpath, f'synthetic_seismic_final_{data_id}.dat')
        label_path = os.path.join(kxpath, f'synthetic_seismic_final_{data_id}.dat')
        if not os.path.isfile(data_path):
            missing_files.append(data_path)
        if not os.path.isfile(label_path):
            missing_files.append(label_path)

    if missing_files:
        preview = '\n'.join(missing_files[:6])
        suffix = '' if len(missing_files) <= 6 else f'\n... and {len(missing_files) - 6} more'
        parser.error(f'Required training files are missing:\n{preview}{suffix}')


validate_data_files(list(tdata_ids) + list(vdata_ids))

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
    criterion = utils.CompositeLoss()


# Load model
if args.model == 'GSCDUnet':
    from Models import GSCDUnet
    model = GSCDUnet.GSCDUnet(input_channels=1, output_channels=1).to(device)
elif args.model == 'UNet':
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
        if 'optimizer_state_dict' in checkpoint:
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
            'best_val_loss': best_val_loss,
            'model_name': args.model,
            'loss_name': args.loss,
            'seed': args.seed
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
            'best_val_loss': best_val_loss,
            'model_name': args.model,
            'loss_name': args.loss,
            'seed': args.seed
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
