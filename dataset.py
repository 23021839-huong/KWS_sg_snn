import torch
import torchaudio
from torch.utils.data import Dataset, WeightedRandomSampler
from collections import Counter
from config import Config
import os

class SpeechCommandsDataset(Dataset):
    def __init__(self, subset="training"):
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=Config.DATA_PATH,
            download=True,
            subset=subset
        )
        # 10 keyword chính + _silence_ + _unknown_
        self.labels = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes', '_silence_', '_unknown_']
        self.label_to_index = {l: i for i, l in enumerate(self.labels)}
        
        # Precompute all labels để tạo sampler
        self._labels = []
        for i in range(len(self.dataset)):
            _, _, label, *_ = self.dataset[i]
            self._labels.append(self.label_to_index.get(label, self.label_to_index['_unknown_']))

    def get_sampler(self):
        """WeightedRandomSampler để cân bằng class imbalance."""
        count = Counter(self._labels)
        # Weight = 1 / frequency của từng class
        weights = [1.0 / count[l] for l in self._labels]
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    def get_class_weights(self):
        """Class weights cho loss function."""
        count = Counter(self._labels)
        total = len(self._labels)
        weights = []
        for i in range(Config.NUM_CLASSES):
            weights.append(total / (Config.NUM_CLASSES * max(count[i], 1)))
        return torch.tensor(weights, dtype=torch.float)

    def __len__(self):
        return len(self.dataset)

    def preprocess(self, waveform, sr):
        if sr != Config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, Config.SAMPLE_RATE)
            waveform = resampler(waveform)

        waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.size(1) > Config.SAMPLE_RATE:
            waveform = waveform[:, :Config.SAMPLE_RATE]
        elif waveform.size(1) < Config.SAMPLE_RATE:
            pad_amount = Config.SAMPLE_RATE - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()
            
        return waveform

    def __getitem__(self, idx):
        waveform, sr, label, *_ = self.dataset[idx]
        waveform = self.preprocess(waveform, sr)
        label_idx = self._labels[idx]
        return waveform, label_idx