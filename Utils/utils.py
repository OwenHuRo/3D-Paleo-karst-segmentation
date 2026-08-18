# utils.py
import numpy as np
from torch import nn
from torch.utils.data import Dataset
import random
import time
import torch
from pathlib import Path

"Generates data for PyTorch"
class DataGenerator(Dataset):
    def __init__(self, dpath, fpath, data_IDs, batch_size=1, dim=(512, 512, 512),
                 n_channels=1, shuffle=True, Enhance=True, SmallScale=False):
        self.dim = dim
        self.dpath = dpath
        self.fpath = fpath
        self.batch_size = batch_size
        self.data_IDs = list(data_IDs)
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.Enhance = Enhance
        self.SmallScale = SmallScale

    def __len__(self):
        return len(self.data_IDs)

    def __getitem__(self, index):
        data_ID = self.data_IDs[index]
        X, Y = self.__data_generation(data_ID)
        return X, Y

    def on_epoch_end(self):
        self.indexes = list(range(len(self.data_IDs)))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, data_ID):
        n1, n2, n3 = self.dim

        if self.SmallScale:
            m1, m2, m3 = 64, 64, 64
        else:
            m1, m2, m3 = 128, 128, 128

        if any(source < crop for source, crop in zip(self.dim, (m1, m2, m3))):
            raise ValueError(
                f"Source volume {self.dim} must be at least as large as "
                f"the requested crop {(m1, m2, m3)}."
            )

        # Memory-map source volumes so only the requested crop is loaded.
        data_path = Path(self.dpath) / f"synthetic_seismic_final_{data_ID}.dat"
        label_path = Path(self.fpath) / f"synthetic_seismic_final_{data_ID}.dat"
        expected_bytes = int(np.prod(self.dim)) * np.dtype(np.float32).itemsize

        for file_path in (data_path, label_path):
            if not file_path.is_file():
                raise FileNotFoundError(f"Volume file not found: {file_path}")
            if file_path.stat().st_size != expected_bytes:
                raise ValueError(
                    f"Invalid file size for {file_path}: expected {expected_bytes} "
                    f"bytes, found {file_path.stat().st_size}."
                )

        gx = np.memmap(data_path, dtype=np.float32, mode='r', shape=self.dim)
        kx = np.memmap(label_path, dtype=np.float32, mode='r', shape=self.dim)

        if self.Enhance:
            # Randomly cut a smaller volume
            k1 = random.randint(0, n1 - m1)
            k2 = random.randint(0, n2 - m2)
            k3 = random.randint(0, n3 - m3)
            gx_cut = gx[k1:k1 + m1, k2:k2 + m2, k3:k3 + m3]
            kx_cut = kx[k1:k1 + m1, k2:k2 + m2, k3:k3 + m3]
            # Normalize the data
            gm = np.mean(gx_cut)
            gs = np.std(gx_cut)
            gx_cut = (gx_cut - gm) / (gs + 1e-8)
            num_rotations = random.randint(0, 3)
            axes = random.choice([(0, 1), (0, 2), (1, 2)])
            gx_cut = np.rot90(gx_cut, num_rotations, axes).copy()
            kx_cut = np.rot90(kx_cut, num_rotations, axes).copy()
        else:
            # Preserve the fixed manuscript evaluation crop while supporting
            # source volumes that have less than 32 voxels of crop margin.
            k1 = min(32, n1 - m1)
            k2 = min(32, n2 - m2)
            k3 = min(32, n3 - m3)
            gx_cut = gx[k1:k1 + m1, k2:k2 + m2, k3:k3 + m3]
            kx_cut = kx[k1:k1 + m1, k2:k2 + m2, k3:k3 + m3]
            gm = np.mean(gx_cut)
            gs = np.std(gx_cut)
            gx_cut = (gx_cut - gm) / (gs + 1e-8)
            kx_cut = np.asarray(kx_cut).copy()

        del gx, kx

        # Add channel dimension
        X = np.expand_dims(gx_cut, axis=0)  # Shape: (1, 128, 128, 128)
        Y = np.expand_dims(kx_cut, axis=0)  # Shape: (1, 128, 128, 128)

        # Convert to PyTorch tensors
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()

        return X, Y


class WeightedBCE(nn.Module):
    def __init__(self, weights=[0.5, 0.5]):
        super(WeightedBCE, self).__init__()
        self.weights = weights

    def forward(self, predict_matrix, truth_matrix):
        predict = predict_matrix.flatten()
        truth = truth_matrix.flatten()
        assert(predict.shape == truth.shape)
        loss = F.binary_cross_entropy(predict, truth, reduction='none')
        pos = (truth > 0.5).float()
        neg = (truth < 0.5).float()
        pos_num = pos.sum().item() + 1e-12
        neg_num = neg.sum().item() + 1e-12
        loss = ((self.weights[0]*pos/pos_num + self.weights[1]*neg/neg_num)*loss).sum()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predict_matrix, truth_matrix):
        bce = F.binary_cross_entropy(predict_matrix, truth_matrix, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = ((truth_matrix*self.alpha+(1-truth_matrix)*(1-self.alpha))*((1-pt)
                        **self.gamma)*bce).mean()
        return focal_loss


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=2, smooth=1e-12):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, predict_matrix, truth_matrix):

        tp = (truth_matrix * predict_matrix).sum(dim=(2, 3, 4))

        fn = ((1 - predict_matrix) * truth_matrix).sum(dim=(2, 3, 4))

        fp = ((1 - truth_matrix) * predict_matrix).sum(dim=(2, 3, 4))

        ti = (tp + self.smooth) / (tp + self.beta * fn + self.alpha * fp + self.smooth)

        ftl = ((1 - ti).pow(self.gamma)).mean()

        return ftl


class AdaptiveTverskyFocalLoss(nn.Module):
    def __init__(self, tversky_weight=0.4, focal_weight=0.6, total_epoch=100):
        super().__init__()
        self.tversky = FocalTverskyLoss()
        self.focal = FocalLoss()
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight
        self.total_epochs = total_epoch

    def forward(self, inputs, targets, epoch):

        tversky_loss = self.tversky(inputs, targets)
        focal_loss = self.focal(inputs, targets)
        tversky_weight, focal_weight = self.get_weights(epoch)
        return tversky_weight*tversky_loss + focal_weight*focal_loss

    def get_weights(self, epoch):
        focal_w = max(self.focal_weight - self.tversky_weight * (epoch / self.total_epochs), self.tversky_weight)
        tversky_w = 1 - focal_w
        return tversky_w, focal_w

class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=[0.9, 0.1]): # W_pos=0.8, W_neg=0.2
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights

    def forward(self, logit, truth, smooth=1e-5):
        batch_size = len(logit)
        logit = logit.flatten()
        truth = truth.flatten()
        assert(logit.shape==truth.shape)
        p = logit.view(batch_size,-1)
        t = truth.view(batch_size,-1)
        w = truth.detach()
        w = w*(self.weights[1]-self.weights[0])+self.weights[0]

        p = w*(p)
        t = w*(t)
        intersection = (p * t).sum(-1)
        union = (p * p).sum(-1) + (t * t).sum(-1)
        dice = 1 - (2*intersection + smooth) / (union +smooth)

        loss = dice.mean()
        return loss


class DWBLossBinary(nn.Module):
    def __init__(self, pos_weight_ratio=0.01):
        super(DWBLossBinary, self).__init__()

        self.w_pos = torch.log(torch.tensor(1.0 / pos_weight_ratio)) + 1  # 正样本权重
        self.w_neg = torch.tensor(1.0)  # 负样本固定权重

    def forward(self, predict_matrix, targets):


        probs = predict_matrix

        ce_term = - (
                self.w_pos * targets * torch.log(probs.clamp(min=1e-7, max=1-1e-7)) +
                self.w_neg * (1 - targets) * torch.log((1 - probs).clamp(min=1e-7, max=1-1e-7))
        )

        brier_term = probs * (1 - probs)

        total_loss = (ce_term + 2 * brier_term).mean()  # 2 times

        return total_loss

import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

def tv_loss(x):
    # x: (B, C, D, H, W)
    dx = torch.abs(x[...,1:,:,:] - x[...,:-1,:,:]).mean()
    dy = torch.abs(x[..., :,1:,:] - x[..., :,:-1,:]).mean()
    dz = torch.abs(x[..., :, :,1:] - x[..., :,:,:-1]).mean()
    return dx + dy + dz


def compute_signed_distance_map(mask_np: np.ndarray) -> np.ndarray:
    """
    caculate 3D signed distance map
    """
    dist_fore = distance_transform_edt(mask_np)
    dist_back = distance_transform_edt(1 - mask_np)
    distance_map = dist_fore.astype(np.float32) - dist_back.astype(np.float32)
    return distance_map


def boundary_weighted_bce_loss(pred: torch.Tensor,
                               target: torch.Tensor,
                               distance_map: torch.Tensor) -> torch.Tensor:

    abs_dist = distance_map.abs()                              # (B, D, H, W)
    max_dist = abs_dist.view(abs_dist.size(0), -1).max(dim=1)[0]  # (B,)
    max_dist = max_dist.view(-1, 1, 1, 1) + 1e-8

    w = 1.0 - (abs_dist / max_dist)    # (B, D, H, W)
    weight_map = 1.0 + w               # (B, D, H, W)
    weight_map = weight_map.unsqueeze(1)  # (B, 1, D, H, W)

    bce = F.binary_cross_entropy(pred, target, reduction='none')  # (B,1,D,H,W)
    loss = (weight_map * bce).mean()
    return loss


def gradient_consistency_loss(pred: torch.Tensor,
                              target: torch.Tensor) -> torch.Tensor:
    """
    Gradient Consistency Loss
    """
    def diff3d(x):
        dx = x[..., 1:, :, :] - x[..., :-1, :, :]
        dy = x[..., :, 1:, :] - x[..., :, :-1, :]
        dz = x[..., :, :, 1:] - x[..., :, :, :-1]
        return dx, dy, dz

    pdx, pdy, pdz = diff3d(pred)
    tdx, tdy, tdz = diff3d(target)

    loss = (pdx - tdx).abs().mean() \
         + (pdy - tdy).abs().mean() \
         + (pdz - tdz).abs().mean()
    return loss


class CompositeLoss(nn.Module):
    """
    Composite loss proposed for GSCD-Unet.

    The loss is the unweighted sum of Focal Loss, Boundary-aware Weighted BCE
    and Gradient Consistency Loss, matching the accompanying manuscript.
    """
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, predict_matrix, truth_matrix):
        focal = self.focal_loss(predict_matrix, truth_matrix)

        masks_np = truth_matrix.detach().cpu().numpy()[:, 0]
        distance_maps_np = np.stack([
            compute_signed_distance_map(mask_np)
            for mask_np in masks_np
        ])
        distance_map = torch.from_numpy(distance_maps_np).to(
            device=predict_matrix.device,
            dtype=predict_matrix.dtype
        )

        boundary = boundary_weighted_bce_loss(
            predict_matrix,
            truth_matrix,
            distance_map
        )
        gradient = gradient_consistency_loss(predict_matrix, truth_matrix)
        return focal + boundary + gradient


def gradient_consistency_loss_mip(pred: torch.Tensor,
                                  target: torch.Tensor) -> torch.Tensor:
    """
    MIP projection distribution consistency loss
    """
    pd = pred.squeeze(1)   # (B, D, H, W)
    td = target.squeeze(1) # (B, D, H, W)

    proj_DH_p, _ = pd.max(dim=3)
    proj_DH_t, _ = td.max(dim=3)
    proj_DW_p, _ = pd.max(dim=2)
    proj_DW_t, _ = td.max(dim=2)
    proj_HW_p, _ = pd.max(dim=1)
    proj_HW_t, _ = td.max(dim=1)


    def diff2d(x: torch.Tensor):
        dx = x[..., 1:, :] - x[..., :-1, :]
        dy = x[..., :, 1:] - x[..., :, :-1]
        return dx, dy


    loss = 0.0
    for p_view, t_view in [(proj_DH_p, proj_DH_t),
                            (proj_DW_p, proj_DW_t),
                            (proj_HW_p, proj_HW_t)]:
        pdx, pdy = diff2d(p_view)
        tdx, tdy = diff2d(t_view)
        loss += (pdx - tdx).abs().mean()
        loss += (pdy - tdy).abs().mean()
    return loss

class WeightedDiceBCE(nn.Module):
    def __init__(self,dice_weight=1,BCE_weight=1):
        super(WeightedDiceBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        inputs[inputs>=0.5] = 1
        inputs[inputs<0.5] = 0
        targets[targets>0] = 1
        targets[targets<=0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE
        return dice_BCE_loss

class EnhancedDiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=0.6, focal_weight=0.4, gamma=2, alpha=0.8):
        super().__init__()
        self.dice = WeightedDiceLoss(weights=[0.5, 0.5])
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        dice_loss = self.dice(inputs, targets)
        focal_loss = self.focal(inputs, targets)
        return self.dice_weight*dice_loss + self.focal_weight*focal_loss

class AdaptiveWeightScheduler:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs

    def get_weights(self, epoch):
        dice_w = max(0.6 - 0.4 * (epoch / self.total_epochs), 0.4)
        focal_w = 1 - dice_w
        return dice_w, focal_w

class PositionAwareEnhancedDiceFocalLoss(nn.Module):
    def __init__(self, position_embed_dim=64, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets, position_embeddings):
        # Dice Loss
        dice_loss = 1 - self.dice_coeff(outputs, targets)

        # Focal Loss
        bce_loss = F.binary_cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        weights = position_embeddings
        b, n_patch, _ = weights.size()
        d, w, h = int(np.cbrt(n_patch)), int(np.cbrt(n_patch)), int(np.cbrt(n_patch))
        weights = weights.permute(0, 2, 1)
        weights = weights.contiguous().view(b, 1, d, h, w)

        weights_upsampled = F.interpolate(
            weights,
            size=(128, 128, 128),
            mode='trilinear',
            align_corners=False
        )  # (B, 1, 128, 128, 128)

        weighted_focal_loss = (focal_loss * weights_upsampled).mean()

        # total
        total_loss = self.alpha * dice_loss + (1 - self.alpha) * weighted_focal_loss
        return total_loss

if __name__ == "__main__":
    """
    loss function test
    """
    input_tensor = torch.tensor(torch.rand((1, 1, 128, 128, 128)), dtype=torch.float)
    target_tensor = torch.tensor(torch.randint(0, 2, (1, 1, 128, 128, 128)), dtype=torch.float)

    weight = torch.tensor([0.8, 0.2],device='cuda')
    Loss = WeightedBCE([0.8, 0.2])
    Loss_Value = Loss(input_tensor, target_tensor)

    print(Loss_Value)
    Loss = FocalLoss()
    Loss_Value = Loss(input_tensor, target_tensor)
    print(Loss_Value)

    Loss = FocalTverskyLoss()
    Loss_Value = Loss(input_tensor, target_tensor)
    print(Loss_Value)

    Loss = AdaptiveTverskyFocalLoss(total_epoch=100)
    Loss_Value = Loss(input_tensor, target_tensor, 10)
    print(Loss_Value)

    Loss = WeightedDiceLoss()
    Loss_Value = Loss(input_tensor, target_tensor)
    print(Loss_Value)

    Loss =DWBLossBinary()
    Loss_Value = Loss(input_tensor, target_tensor)
    print(Loss_Value)

    start_time = time.time()
    Loss_Value = gradient_consistency_loss_mip(input_tensor, target_tensor)
    end_time = time.time()
    print("cost: {:.4f}s".format(end_time - start_time))

    start_time = time.time()
    Loss_Value1 = gradient_consistency_loss(input_tensor, target_tensor)
    end_time = time.time()
    print("cost: {:.4f}s".format(end_time - start_time))
