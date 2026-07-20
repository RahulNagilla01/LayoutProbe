import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.models import vit_b_16, ViT_B_16_Weights
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from dataset import LayoutProbeDataset

SEED       = 42
BATCH_SIZE = 64
EPOCHS     = 10
LR         = 2e-5
VAL_SPLIT  = 0.2
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH = "results/best_vit.pth"
PLOT_SAVE_PATH  = "results/finetune_curve.png"


def get_dataloaders():
    train_ds   = LayoutProbeDataset(split="train")
    val_size   = int(len(train_ds) * VAL_SPLIT)
    train_size = len(train_ds) - val_size
    generator  = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(train_ds, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, train_ds


def build_model(num_classes):
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads = nn.Linear(768, num_classes) #custom layout classes head
    return model


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = correct = total = 0
    for imgs, layout_labels, _ in loader:
        imgs, layout_labels = imgs.to(DEVICE), layout_labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, layout_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (logits.argmax(dim=1) == layout_labels).sum().item()
        total      += imgs.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for imgs, layout_labels, _ in loader:
            imgs, layout_labels = imgs.to(DEVICE), layout_labels.to(DEVICE)
            logits     = model(imgs)
            total_loss += criterion(logits, layout_labels).item()
            correct    += (logits.argmax(dim=1) == layout_labels).sum().item()
            total      += imgs.size(0)
    return total_loss / len(loader), correct / total


def save_plot(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses,   label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.legend()
    ax2.plot(train_accs, label="Train Acc")
    ax2.plot(val_accs,   label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curve")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH)
    plt.close()


def finetune():
    torch.manual_seed(SEED)
    os.makedirs("results", exist_ok=True)

    train_loader, val_loader, train_ds = get_dataloaders()
    num_classes   = train_ds.num_layout_classes()
    class_weights = train_ds.class_weights().to(DEVICE)

    model     = build_model(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights) #using CEL 
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

  
    print(f"Train crops    : {len(train_loader.dataset)}")
    print(f"Val crops      : {len(val_loader.dataset)}")
    print(f"Layout classes : {num_classes}")
  

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH) #only saving check points 
            

    save_plot(train_losses, val_losses, train_accs, val_accs)
    print(f"Best val loss:{best_val_loss:.3f}")


if __name__ == "__main__":
    finetune()