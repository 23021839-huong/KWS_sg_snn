import torch
import torchaudio
from config import Config

class LogMelTransform:
    def __init__(self):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_mels=Config.N_MELS,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH
        )

    def __call__(self, waveform):
        """
        waveform có thể là:
        (samples)
        (1, samples)
        """

        # 👉 đảm bảo shape (1, samples)
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

        # 👉 luôn squeeze về 2D
        log_mel = log_mel.squeeze(0)   # (n_mels, time)

        # 👉 đảm bảo đúng 2D trước permute
        if log_mel.dim() != 2:
            raise ValueError(f"log_mel sai shape: {log_mel.shape}")

        log_mel = log_mel.permute(1, 0)  # (time, feature)

        # normalize
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)

        return log_mel


def pad(x):
    if x.shape[0] > Config.MAX_LEN:
        return x[:Config.MAX_LEN]
    else:
        pad_len = Config.MAX_LEN - x.shape[0]
        pad = torch.zeros(pad_len, x.shape[1])
        return torch.cat([x, pad], dim=0)