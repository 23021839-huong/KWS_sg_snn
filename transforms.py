import torch
import torchaudio
import torchaudio.transforms as T
from config import Config


class LogMelTransform:
    def __init__(self, augment=False):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_mels=Config.N_MELS,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH
        )
        self.augment = augment

        # SpecAugment — chỉ dùng khi training
        if augment:
            self.time_mask = T.TimeMasking(time_mask_param=10)
            self.freq_mask = T.FrequencyMasking(freq_mask_param=8)

    def __call__(self, waveform):
        """
        waveform có thể là:
        (samples), (1, samples), (1, 1, samples)
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        elif waveform.dim() == 2:
            pass
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            raise ValueError(f"Waveform shape sai: {waveform.shape}")

        mel = self.mel(waveform)   # (1, n_mels, time)
        log_mel = torch.log(mel + 1e-9)

        # SpecAugment trước khi squeeze (cần shape 3D)
        if self.augment:
            log_mel = self.time_mask(log_mel)
            log_mel = self.freq_mask(log_mel)

        log_mel = log_mel.squeeze(0)   # (n_mels, time)

        if log_mel.dim() != 2:
            raise ValueError(f"log_mel sai shape: {log_mel.shape}")

        log_mel = log_mel.permute(1, 0)  # (time, feature)

        # Normalize per-sample
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)

        return log_mel


def pad(x):
    """Pad hoặc truncate về MAX_LEN. Giữ nguyên device của tensor."""
    if x.shape[0] > Config.MAX_LEN:
        return x[:Config.MAX_LEN]
    else:
        pad_len = Config.MAX_LEN - x.shape[0]
        # Fix: dùng device=x.device để tránh crash khi x ở GPU
        pad_tensor = torch.zeros(pad_len, x.shape[1], device=x.device)
        return torch.cat([x, pad_tensor], dim=0)