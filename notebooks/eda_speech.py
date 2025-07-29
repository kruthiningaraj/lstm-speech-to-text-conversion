import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def eda_speech(audio_path):
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)
    print(f"Sample rate: {sr}, Duration: {len(y)/sr:.2f} seconds")

    # Plot waveform
    plt.figure(figsize=(14,4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Audio Waveform')
    plt.show()

    # Plot spectrogram
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    plt.figure(figsize=(10,4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()

    # Plot MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10,4))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title('MFCCs')
    plt.show()

if __name__ == "__main__":
    eda_speech("data/train/audio/sample.wav")  # Replace with actual path
