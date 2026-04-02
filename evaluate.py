import torch
from config import Config
from transforms import LogMelTransform, pad
from encoding import rate_encoding

transform = LogMelTransform()

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for waveform, labels in loader:
            labels = labels.to(Config.DEVICE)

            features_list = []

            for w in waveform:
                f = transform(w.unsqueeze(0))
                f = pad(f)
                features_list.append(f)

            features = torch.stack(features_list)
            features = features.to(Config.DEVICE)

            spikes = rate_encoding(features)
            output = model(spikes)

            spike_count = output.sum(dim=0)

            preds = spike_count.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total