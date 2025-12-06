import torch.nn as nn
try:
    from torchvision.models import efficientnet_b0
except Exception:
    from torchvision.models import efficientnet_b0

def build_model(num_classes):
    model = efficientnet_b0(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
