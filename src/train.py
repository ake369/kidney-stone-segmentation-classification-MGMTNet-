
import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import KidneyStoneDataset
from model import MGMTNet

# Optimize CPU multi-threading performance for Intel processors
torch.set_num_threads(8)


# =========================================================================
# COMBINED SEGMENTATION LOSS (BCE + DICE)
# =========================================================================
class CombinedSegLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        
        # Flatten tensors for Dice calculation
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice_loss = 1.0 - ((2.0 * intersection + self.smooth) / 
                           (probs_flat.sum() + targets_flat.sum() + self.smooth))
        return bce_loss + dice_loss


# =========================================================================
# TRAINING ENGINE
# =========================================================================
def train_one_epoch(model, dataloader, optimizer, seg_criterion, cls_criterion, scaler, device, alpha, beta, gamma, epoch_num, total_epochs):
    model.train()
    total_loss = 0.0
    total_k_loss, total_s_loss, total_c_loss = 0.0, 0.0, 0.0
    correct, total_samples = 0, 0

    use_amp = (device.type == 'cuda')

    # Add tqdm progress bar for training
    pbar = tqdm(dataloader, desc=f"Epoch [{epoch_num:03d}/{total_epochs:03d}] Train", leave=False)

    for batch in pbar:
        images = batch['image'].to(device)
        k_masks = batch['kidney_mask'].to(device)
        s_masks = batch['stone_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            
            # Compute sub-losses
            loss_k = seg_criterion(outputs['kidney_logits'], k_masks)
            loss_s = seg_criterion(outputs['stone_logits'], s_masks)
            loss_c = cls_criterion(outputs['cls_logits'], labels)
            
            # Multi-task loss formulation
            loss = (alpha * loss_k) + (beta * loss_s) + (gamma * loss_c)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Metrics Tracking
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_k_loss += loss_k.item() * bs
        total_s_loss += loss_s.item() * bs
        total_c_loss += loss_c.item() * bs

        preds = torch.argmax(outputs['cls_logits'], dim=1)
        correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        # Update progress bar description with current batch loss
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    n = len(dataloader.dataset)
    return total_loss / n, total_k_loss / n, total_s_loss / n, total_c_loss / n, (correct / total_samples) * 100.0


@torch.no_grad()
def evaluate(model, dataloader, seg_criterion, cls_criterion, device, alpha, beta, gamma, epoch_num, total_epochs):
    model.eval()
    total_loss = 0.0
    correct, total_samples = 0, 0
    use_amp = (device.type == 'cuda')

    # Add tqdm progress bar for validation
    pbar = tqdm(dataloader, desc=f"Epoch [{epoch_num:03d}/{total_epochs:03d}] Val  ", leave=False)

    for batch in pbar:
        images = batch['image'].to(device)
        k_masks = batch['kidney_mask'].to(device)
        s_masks = batch['stone_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss_k = seg_criterion(outputs['kidney_logits'], k_masks)
            loss_s = seg_criterion(outputs['stone_logits'], s_masks)
            loss_c = cls_criterion(outputs['cls_logits'], labels)
            
            loss = (alpha * loss_k) + (beta * loss_s) + (gamma * loss_c)

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs['cls_logits'], dim=1)
        correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})

    n = len(dataloader.dataset)
    return total_loss / n, (correct / total_samples) * 100.0
# =========================================================================
# MAIN PIPELINE
# =========================================================================
def main():
    # 1. Configuration & Hyperparameters
    IMG_DIR_TRAIN = "./data/train/images"
    MSK_DIR_TRAIN = "./data/train/masks"
    IMG_DIR_VAL   = "./data/val/images"
    MSK_DIR_VAL   = "./data/val/masks"
    
    TARGET_SIZE = (512, 512)
    BATCH_SIZE = 16
    EPOCHS = 50
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    
    # Loss Coefficients
    ALPHA = 1.0  # Kidney segmentation weight
    BETA  = 1.5  # Stone segmentation weight
    GAMMA = 0.8  # Classification weight

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on compute device: {device}")

    # Set parameters safely according to device
    is_cuda = (device.type == 'cuda')
    num_workers = 4 if is_cuda else 0
    pin_memory = is_cuda

    # 2. Datasets and Loaders
    train_dataset = KidneyStoneDataset(IMG_DIR_TRAIN, MSK_DIR_TRAIN, target_size=TARGET_SIZE)
    val_dataset   = KidneyStoneDataset(IMG_DIR_VAL, MSK_DIR_VAL, target_size=TARGET_SIZE)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    # 3. Model Initialization
    model = MGMTNet(in_channels=3, num_classes=2).to(device)

    # 4. Optimizer, Scheduler, and Losses
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    warmup_epochs = 5
    lr_lambda = lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else \
                0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    seg_criterion = CombinedSegLoss().to(device)
    cls_criterion = nn.CrossEntropyLoss().to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=is_cuda)

    # 5. Training Loop
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_k_loss, train_s_loss, train_c_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, seg_criterion, cls_criterion, scaler, device, ALPHA, BETA, GAMMA, epoch, EPOCHS
        )
        val_loss, val_acc = evaluate(
            model, val_loader, seg_criterion, cls_criterion, device, ALPHA, BETA, GAMMA, epoch, EPOCHS
        )
        
        scheduler.step()

        print(f"Epoch [{epoch:03d}/{EPOCHS:03d}] | "
              f"Train Loss: {train_loss:.4f} (K:{train_k_loss:.3f}, S:{train_s_loss:.3f}, C:{train_c_loss:.3f}) | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save Best Model Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/model_best.pth")
            print(f"  --> Saved new best checkpoint with Val Acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
