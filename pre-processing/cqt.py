import librosa
import numpy as np
import matplotlib.pyplot as plt


file_spoofed = "../asvspoof5/flac_T/T_0000000000.flac"
file_bonafide = "../asvspoof5/flac_T/T_0000000011.flac"
y, sr = librosa.load(file_spoofed)
y2, sr2 = librosa.load(file_bonafide)
# 3. Run the default beat tracker
# tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
# print(f"Estimated tempo: {tempo[0]:.2f} beats per minute"

# Compute CQT
C = librosa.cqt(
    y,
    sr=sr,
    hop_length=512,
    n_bins=84,
    bins_per_octave=12
)

# Convert to magnitude (important!)
C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)

# Visualize
plt.figure(figsize=(10, 4))
librosa.display.specshow(C_db, sr=sr, x_axis='time', y_axis='cqt_note')
plt.colorbar(format='%+2.0f dB')
plt.title('CQT Spectrogram')
plt.tight_layout()
plt.show()

C = librosa.cqt(
    y2,
    sr=sr2,
    hop_length=512,
    n_bins=84,
    bins_per_octave=12
)

# Convert to magnitude (important!)
C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)

# Visualize
plt.figure(figsize=(10, 4))
librosa.display.specshow(C_db, sr=sr2, x_axis='time', y_axis='cqt_note')
plt.colorbar(format='%+2.0f dB')
plt.title('CQT Spectrogram')
plt.tight_layout()
plt.show()
