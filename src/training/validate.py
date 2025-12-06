import torch
from tqdm import tqdm
from src.utils.metrics import accuracy_score, macro_f1

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, targets = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())

    acc = accuracy_score(targets, preds)
    f1 = macro_f1(targets, preds)

    return total_loss / len(loader), acc, f1
