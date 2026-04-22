import torch

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    DATA_PATH = "./data"

    SAMPLE_RATE = 16000
    N_MELS = 40
    N_FFT = 400
    HOP_LENGTH = 160
    MAX_LEN = 64

    NUM_CLASSES = 12
    BETA = 0.95

    BATCH_SIZE = 64       
    LR = 1e-3            
    EPOCHS = 30           
    WEIGHT_DECAY = 1e-3  

    TIME_STEPS = 50    