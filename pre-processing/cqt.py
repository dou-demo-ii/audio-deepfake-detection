import librosa
import numpy as np
import matplotlib.pyplot as plt

train = open("../asvspoof5/ASVspoof5_protocols/ASVspoof5.train.tsv", "r")
lines = train.readlines()
train.close()

spoofed = True
data_count = 0
get_data = 100
spoofed_files = []
bonafide_files = []
i = 0
while data_count < get_data :
    if "spoof" in lines[i] and spoofed:
        spoofed_files.append(lines[i].strip().split()[1] + ".flac")
        spoofed = not spoofed
        data_count += 1
    elif "bonafide" in lines[i] and not spoofed:
        bonafide_files.append(lines[i].strip().split()[1] + ".flac")
        spoofed = not spoofed
        data_count += 1
    i += 1

print(spoofed_files)
print(bonafide_files)

def preprocess(filenames, title):
    res = {}
    for filename in filenames:
        y, sr = librosa.load(f"../asvspoof5/flac_T/{filename}", sr=16000)

        # preemphasis untuk frekuensi tinggi dimana itu
        y = librosa.effects.preemphasis(y, coef=0.97, zi=None, return_zf=False)

        C = librosa.cqt(
            y,
            sr=sr,
            hop_length=512,
            n_bins=84,
            bins_per_octave=12
        )

        # convert to magnitudde
        res[filename] = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        save_path = f"{title}/{filename[:-5]}.npy"
        open(save_path, 'x')
        np.save(save_path, res[filename])
    # with open(f"storage/{title}.txt", "x") as f:
    #     f.write(str(res))
    return res

preprocessed_spoofed = preprocess(spoofed_files, "spoofed")
preprocessed_bonafide = preprocess(bonafide_files, "bonafide")
