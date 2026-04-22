import torch
from config import Config
from encoding import rate_encoding


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for features, labels in loader:
            # features: (B, time, feature) — đã được transform trong Dataset
            features = features.to(Config.DEVICE)
            labels   = labels.to(Config.DEVICE)

            spikes = rate_encoding(features)       # (T, B, time, feature)
            output = model(spikes)                 # (T, B, num_classes)
            spike_count = output.sum(dim=0)        # (B, num_classes)

            preds = spike_count.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total