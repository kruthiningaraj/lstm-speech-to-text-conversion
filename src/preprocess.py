import os
import librosa
import numpy as np

def extract_features(audio_path, sr=16000, n_mfcc=40):
    y, sr = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T

def load_dataset(audio_dir, transcript_file):
    features, transcripts = [], []
    with open(transcript_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split('\t')
        audio_file, text = parts[0], parts[1]
        mfccs = extract_features(os.path.join(audio_dir, audio_file))
        features.append(mfccs)
        transcripts.append(text)
    return features, transcripts
