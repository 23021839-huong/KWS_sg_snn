import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from config import Config


class KWS_SNN(nn.Module):
    def __init__(self):
        super().__init__()

        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Conv block 1
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)

        # Conv block 2
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        # Tính flatten size tự động
        with torch.no_grad():
            dummy = torch.zeros(1, 1, Config.MAX_LEN, Config.N_MELS)
            out = self.pool1(torch.relu(self.bn1(self.conv1(dummy))))
            out = self.pool2(torch.relu(self.bn2(self.conv2(out))))
            self.flatten_size = out.view(1, -1).shape[1]

        print(f"[*] Flatten size: {self.flatten_size}")

        self.dropout = nn.Dropout(0.5)   # Tăng từ 0.3 → giảm overfitting

        self.fc1  = nn.Linear(self.flatten_size, 256)
        self.lif1 = snn.Leaky(beta=Config.BETA, spike_grad=spike_grad, learn_beta=True)

        self.fc2  = nn.Linear(256, 128)
        self.lif2 = snn.Leaky(beta=Config.BETA, spike_grad=spike_grad, learn_beta=True)

        self.fc3  = nn.Linear(128, Config.NUM_CLASSES)
        self.lif3 = snn.Leaky(beta=Config.BETA, spike_grad=spike_grad, learn_beta=True)

    def forward(self, x):
        """
        x: (T, B, time, feature)

        CNN chỉ chạy 1 lần trên mean spike input → nhanh hơn ~T lần.
        Chỉ các SNN layers (FC + LIF) mới lặp theo time steps.
        """
        T_steps = x.size(0)

        # --- CNN: chạy 1 lần ---
        # Dùng mean theo thời gian để có signal ổn định
        x_mean = x.mean(dim=0).unsqueeze(1)   # (B, 1, time, feature)
        feat = self.pool1(torch.relu(self.bn1(self.conv1(x_mean))))
        feat = self.pool2(torch.relu(self.bn2(self.conv2(feat))))
        feat = feat.reshape(feat.size(0), -1)  # (B, flatten_size)

        # --- SNN: lặp T_steps lần ---
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        outputs = []

        for t in range(T_steps):
            cur = self.dropout(feat)

            cur = self.fc1(cur)
            spk1, mem1 = self.lif1(cur, mem1)

            cur = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur, mem2)

            cur = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur, mem3)

            outputs.append(spk3)

        return torch.stack(outputs)  # (T, B, num_classes)