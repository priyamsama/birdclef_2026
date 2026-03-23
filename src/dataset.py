import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset


SR = 32000
DURATION = 5
SAMPLES = SR * DURATION  # 160000 samples


def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    # Pad if shorter than 5 seconds
    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES - len(y)))
    # Random crop if longer than 5 seconds
    else:
        start = np.random.randint(0, len(y) - SAMPLES + 1)
        y = y[start:start + SAMPLES]
    return y


def audio_to_melspec(y):
    S = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=128,
        hop_length=320,
        n_fft=1024,
        fmin=50,
        fmax=14000
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
    
    # Fix to exact width of 500
    if S_norm.shape[1] > 500:
        S_norm = S_norm[:, :500]
    elif S_norm.shape[1] < 500:
        S_norm = np.pad(S_norm, ((0, 0), (0, 500 - S_norm.shape[1])))
    
    return S_norm.astype(np.float32)


class BirdDataset(Dataset):
    def __init__(self, df, species_list, audio_dir, is_train=True):
        # Filter low quality ratings for training
        if is_train:
            self.df = df[df["rating"] >= 3].reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)
        self.species_list = species_list
        self.audio_dir = audio_dir
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load and convert audio
        path = os.path.join(self.audio_dir, row["filename"])
        y = load_audio(path)
        spec = audio_to_melspec(y)

        # Shape: (1, 128, 500) — 1 channel like grayscale image
        spec = torch.tensor(spec).unsqueeze(0)

        # Build 234-dim label vector
        label = torch.zeros(len(self.species_list))
        if row["primary_label"] in self.species_list:
            label[self.species_list.index(row["primary_label"])] = 1.0

        return spec, label


if __name__ == "__main__":
    # Quick test — run with: python src/dataset.py
    DATA_DIR = "data"
    AUDIO_DIR = os.path.join(DATA_DIR, "train_audio")

    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    taxonomy = pd.read_csv(os.path.join(DATA_DIR, "taxonomy.csv"))
    species_list = taxonomy["primary_label"].tolist()

    print(f"Total clips: {len(df)}")
    print(f"Species: {len(species_list)}")

    dataset = BirdDataset(df, species_list, AUDIO_DIR, is_train=True)
    print(f"Filtered dataset size: {len(dataset)}")

    spec, label = dataset[0]
    print(f"Spectrogram shape: {spec.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Label sum (should be 1.0): {label.sum()}")
    print("dataset.py OK!")