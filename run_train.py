import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from src.datasets.medical_images import MedicalImageDataset, get_transforms
from src.models.classifier import build_model
from src.training.train import train_one_epoch
from src.training.validate import validate
from src.utils.config import Config

def main():
    cfg = Config()

    transforms = get_transforms(cfg.IMG_SIZE)

    train_ds = MedicalImageDataset("data/train.csv", "data/raw", transforms["train"])
    val_ds = MedicalImageDataset("data/val.csv", "data/raw", transforms["val"])

    train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE)

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    model = build_model(cfg.NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)

    for epoch in range(cfg.EPOCHS):
        print(f"Epoch {epoch+1}/{cfg.EPOCHS}")

        train_loss = train_one_epoch(model, train_dl, criterion, optimizer, device)
        val_loss, acc, f1 = validate(model, val_dl, criterion, device)

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss:   {val_loss:.4f}")
        print(f"Accuracy:   {acc:.4f}")
        print(f"F1-macro:   {f1:.4f}")

        torch.save(model.state_dict(), f"models/checkpoints/epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()
