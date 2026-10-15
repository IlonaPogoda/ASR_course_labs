import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--clean", required=True)
parser.add_argument("--augmented", required=True)
parser.add_argument("--name", default="augmentation")
args = parser.parse_args()

sr = 16000

clean, _ = librosa.load(args.clean, sr=sr)
aug, _ = librosa.load(args.augmented, sr=sr)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for row, (audio, label) in enumerate([
    (clean, "Clean"),
    (aug, args.name)
]):
    t = np.arange(len(audio)) / sr

    axes[row, 0].plot(t, audio, linewidth=0.5)
    axes[row, 0].set_title(f"{label}: waveform")
    axes[row, 0].set_xlabel("Time (s)")
    axes[row, 0].set_ylabel("Amplitude")
    axes[row, 0].set_ylim(-1.1, 1.1)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=400,
        hop_length=160,
        n_mels=80
    )

    logmel = librosa.power_to_db(mel, ref=1.0)

    librosa.display.specshow(
        logmel,
        sr=sr,
        hop_length=160,
        x_axis="time",
        y_axis="mel",
        ax=axes[row, 1],
        vmin=-80,
        vmax=0
    )

    axes[row, 1].set_title(f"{label}: log-Mel")

plt.tight_layout()
plt.savefig(f"outputs/compare_{args.name}.png", dpi=150)
plt.show()