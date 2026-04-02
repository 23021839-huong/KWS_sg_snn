# Keyword Spotting using Spiking Neural Network (KWS-SNN)

## 1. Introduction

This project implements a Keyword Spotting (KWS) system using Spiking Neural Networks (SNN). The goal is to recognize short spoken commands from audio signals. The model is trained on the SpeechCommands dataset and uses biologically inspired spiking neurons instead of traditional artificial neurons.

The system combines signal processing, deep learning, and neuromorphic computing techniques.

---

## 2. Objectives

* Build a keyword recognition system using SNN
* Apply spike-based encoding for audio features
* Handle class imbalance in speech datasets
* Evaluate performance on test data
* Provide a complete training and evaluation pipeline

---

## 3. Dataset

The project uses the SpeechCommands dataset provided by Google.

Dataset characteristics:

* Audio sampling rate: 16 kHz
* Short audio clips (~1 second)
* Multiple keyword classes

The dataset is automatically downloaded using torchaudio:

* No manual download is required
* Data is not included in the repository

---

## 4. Data Processing Pipeline

### 4.1 Preprocessing

* Resampling to 16 kHz
* Mono channel conversion
* Padding or trimming to fixed length
* Amplitude normalization

### 4.2 Feature Extraction

* Log-Mel Spectrogram
* Parameters:

  * Number of Mel filters: 40
  * FFT size: 400
  * Hop length: 160

### 4.3 Spike Encoding

* Rate encoding is used
* Features are normalized to [0, 1]
* Converted into spike trains over multiple time steps

---

## 5. Model Architecture

The model is a hybrid CNN + SNN architecture:

### Convolutional Layers

* Conv2D + BatchNorm + ReLU + MaxPooling
* Extract spatial features from spectrogram

### Fully Connected Layers with Spiking Neurons

* Linear layers followed by Leaky Integrate-and-Fire (LIF) neurons
* Surrogate gradient used for training

### Output

* Spike counts accumulated over time
* Final classification based on highest spike activity

---

## 6. Training Strategy

### Loss Function

* CrossEntropyLoss
* Supports class weighting for imbalance handling

### Optimization

* Adam optimizer
* Learning rate scheduling using CosineAnnealingLR

### Regularization

* Dropout
* Weight decay

### Stability Techniques

* Gradient clipping to prevent exploding gradients

### Class Imbalance Handling

* WeightedRandomSampler
* Class-weighted loss

---

## 7. Evaluation

* Model evaluated on test dataset
* Prediction based on spike count aggregation
* Accuracy used as main metric

---

## 8. Project Structure

```
kws_snn/
│
├── config.py        # Hyperparameters and configuration
├── dataset.py       # Dataset loading and preprocessing
├── transforms.py    # Feature extraction (Log-Mel)
├── encoding.py      # Spike encoding
├── model.py         # SNN model definition
├── trainer.py       # Training loop
├── evaluate.py      # Evaluation logic
├── main.py          # Entry point
├── checkpoints/     # Saved models (ignored)
├── data/            # Dataset (ignored)
```

---

## 9. Installation

### Requirements

* Python 3.8+
* PyTorch
* torchaudio
* snntorch

Install dependencies:

```
pip install torch torchaudio snntorch
```

---

## 10. Usage

### Training

```
python main.py
```

### Output

* Training logs (loss, accuracy)
* Best model saved in:

```
checkpoints/best.pth
```

---

## 11. Key Features

* End-to-end KWS pipeline
* Spike-based neural computation
* Efficient handling of imbalanced data
* Modular and extensible code structure
* Automatic dataset handling

---

## 12. Possible Improvements

* Reduce number of classes (e.g., 12 to 5 keywords)
* Real-time inference using microphone input
* Compare SNN with CNN baseline
* Deploy on embedded or edge devices
* Optimize latency and energy efficiency

---

## 13. Conclusion

This project demonstrates how Spiking Neural Networks can be applied to audio classification tasks. It provides a solid foundation for further research in neuromorphic computing and real-time speech recognition systems.
