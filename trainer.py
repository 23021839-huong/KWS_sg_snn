import torch
import torch.nn as nn
from config import Config
from encoding import rate_encoding


def train_epoch(model, loader, optimizer, class_weights=None):
    model.train()

    if class_weights is not None:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(Config.DEVICE))
    else:
        loss_fn = nn.CrossEntropyLoss()

    total_loss, correct, total = 0, 0, 0

    for features, labels in loader:
        # features: (B, time, feature) — đã được transform trong Dataset
        features = features.to(Config.DEVICE)
        labels   = labels.to(Config.DEVICE)

        spikes = rate_encoding(features)       # (T, B, time, feature)
        output = model(spikes)                 # (T, B, num_classes)
        spike_count = output.sum(dim=0)        # (B, num_classes)

        loss = loss_fn(spike_count, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = spike_count.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total