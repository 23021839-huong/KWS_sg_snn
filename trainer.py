import torch
import torch.nn as nn
from config import Config
from transforms import LogMelTransform, pad
from encoding import rate_encoding

transform = LogMelTransform()

def train_epoch(model, loader, optimizer, class_weights=None):
    model.train()

    if class_weights is not None:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(Config.DEVICE))
    else:
        loss_fn = nn.CrossEntropyLoss()

    total_loss, correct, total = 0, 0, 0

    for waveform, labels in loader:
        labels = labels.to(Config.DEVICE)

        features_list = []
        for w in waveform:
            f = transform(w)
            f = pad(f)
            features_list.append(f)

        features = torch.stack(features_list).to(Config.DEVICE)  # (B, time, feature)
        spikes = rate_encoding(features)                          # (T, B, time, feature)

        output = model(spikes)          # (T, B, num_classes)
        spike_count = output.sum(dim=0) # (B, num_classes)

        loss = loss_fn(spike_count, labels)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — tránh gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = spike_count.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total