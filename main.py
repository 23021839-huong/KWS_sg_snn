import torch
from torch.utils.data import DataLoader
from config import Config
from dataset import SpeechCommandsDataset
from model import KWS_SNN
from trainer import train_epoch
from evaluate import evaluate
import os

if __name__ == '__main__':
    os.makedirs("checkpoints", exist_ok=True)

    # Dataset — transform đã được nhúng vào Dataset
    train_set = SpeechCommandsDataset("training")
    val_set   = SpeechCommandsDataset("validation")
    test_set  = SpeechCommandsDataset("testing")

    sampler = train_set.get_sampler()
    class_weights = train_set.get_class_weights()

    train_loader = DataLoader(
        train_set,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_set,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_set,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    model = KWS_SNN().to(Config.DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-5
    )

    print(f"Training on : {Config.DEVICE}")
    print(f"Train        : {len(train_set)} samples")
    print(f"Validation   : {len(val_set)} samples")
    print(f"Test         : {len(test_set)} samples")

    best_acc = 0
    for epoch in range(Config.EPOCHS):
        loss, train_acc = train_epoch(model, train_loader, optimizer, class_weights)
        val_acc  = evaluate(model, val_loader)
        scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch+1:02d} | "
            f"Loss={loss:.4f} | "
            f"TrainAcc={train_acc:.4f} | "
            f"ValAcc={val_acc:.4f} | "
            f"LR={lr_now:.2e}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/best.pth")
            print(f"  --> Saved best model (ValAcc={best_acc:.4f})")

    # Đánh giá cuối cùng trên test set
    model.load_state_dict(torch.load("checkpoints/best.pth"))
    test_acc = evaluate(model, test_loader)
    print(f"\nBest ValAcc  : {best_acc:.4f}")
    print(f"Test Acc     : {test_acc:.4f}")