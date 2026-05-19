import os
import random
import warnings

import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew

warnings.filterwarnings("ignore")

# 1. KONFIGURASI PATH & PARAMETER
PROTOCOL_PATH = "../asvspoof5/ASVspoof5_protocols/ASVspoof5.train.tsv"
DATA_DIR = "../asvspoof5/flac_T/"  # Folder dataset training FLAC
SAMPLE_RATE = 16000  # Standar ASVspoof 5 [3]
NUM_SAMPLES_PER_CLASS = 500  # Target 500:500 [1]
PRE_EMPH_COEF = 0.97  # Sesuai Laporan 4 Bab III.3 [4]


# 2. SAMPLING DATA BERDASARKAN PROTOKOL [5]
def get_balanced_subset(protocol_path, n_samples):
    df_proto = pd.read_csv(
        protocol_path,
        sep=r"\s+",
        header=None,
        names=["SID", "FID", "GENDER", "CODEC", "Q", "S", "TAG", "LABEL", "KEY", "TMP"],
    )

    # NEW: Filter dataset to only include FIDs from T_0000000000 to T_0000036499
    df_proto = df_proto[df_proto["FID"].between("T_0000000000", "T_0000036499")]

    # Ambil list file bona fide dan spoof dari partisi yang tersisa
    bf_files = df_proto[df_proto["KEY"] == "bonafide"]["FID"].tolist()
    spoof_files = df_proto[df_proto["KEY"] == "spoof"]["FID"].tolist()

    # SAFETY NET: Pastikan partisi ini memiliki cukup file untuk target 500
    actual_bf = len(bf_files)
    actual_spoof = len(spoof_files)

    if actual_bf < n_samples or actual_spoof < n_samples:
        n_samples = min(actual_bf, actual_spoof, n_samples)
        print(
            f"⚠️ Peringatan: Partisi tidak memiliki {NUM_SAMPLES_PER_CLASS} sampel per kelas."
        )
        print(
            f"Menyesuaikan target sampling menjadi: {n_samples} per kelas (BF: {actual_bf}, Spoof: {actual_spoof})"
        )

    # Sampling acak
    selected_bf = random.sample(bf_files, n_samples)
    selected_spoof = random.sample(spoof_files, n_samples)

    return selected_bf, selected_spoof


# 3. FUNGSI EKSTRAKSI FITUR (UTTERANCE-LEVEL SCALARS)
def extract_file_stats(file_id, label):
    path = os.path.join(DATA_DIR, f"{file_id}.flac")
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        duration = librosa.get_duration(y=y, sr=sr)

        # A. Pre-emphasis [4, 6]
        y_pre = librosa.effects.preemphasis(y, coef=PRE_EMPH_COEF)

        # F0/Pitch (F12) menggunakan pYIN
        # fmin dan fmax disesuaikan dengan rentang vokal manusia (C2 - C7)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y_pre, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
        )
        # Ambil rata-rata hanya dari frame yang 'voiced' (abaikan NaN)
        f0_mean = np.nanmean(f0) if not np.all(np.isnan(f0)) else 0

        # B. LFCC (60 coeff) - Ambil mean dari seluruh frame [2, 7]
        # Menggunakan librosa.feature.mfcc dengan filter linear sebagai representasi LFCC
        from spafe.features.lfcc import lfcc

        lfcc_features = lfcc(y_pre, fs=sr, num_ceps=60, nfilts=128)
        lfcc_mean = np.mean(lfcc_features)

        # C. CQT Magnitude (avg dB) [7, 8]
        cqt = np.abs(librosa.cqt(y_pre, sr=sr, n_bins=84))
        cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
        cqt_mean = np.mean(cqt_db)

        # D. RMS Energy [9, 10]
        rms = librosa.feature.rms(y=y_pre)
        rms_mean = np.mean(rms)

        # E. Zero Crossing Rate [9, 10]
        zcr = librosa.feature.zero_crossing_rate(y_pre)
        zcr_mean = np.mean(zcr)

        # F. Spectral Centroid [9, 11]
        centroid = librosa.feature.spectral_centroid(y=y_pre, sr=sr)
        centroid_mean = np.mean(centroid)

        # G. Spectral Rolloff (F11 - Rekomendasi tambahan) [2]
        rolloff = librosa.feature.spectral_rolloff(y=y_pre, sr=sr, roll_percent=0.85)
        rolloff_mean = np.mean(rolloff)

        return {
            "Label": label,
            "LFCC_Mean": lfcc_mean,
            "CQT_dB_Mean": cqt_mean,
            "RMS_Mean": rms_mean,
            "ZCR_Mean": zcr_mean,
            "Centroid_Hz": centroid_mean,
            "Rolloff_Hz": rolloff_mean,
            "F0_mean": f0_mean,
            "Duration": duration,
        }
    except Exception as e:
        print(f"Error processing {file_id}: {e}")
        return None


# 4. EKSEKUSI DATA COLLECTION
bf_list, spoof_list = get_balanced_subset(PROTOCOL_PATH, NUM_SAMPLES_PER_CLASS)
all_results = []

print("Mengekstrak fitur Bona Fide...")
for fid in bf_list:
    res = extract_file_stats(fid, "Bona Fide")
    if res:
        all_results.append(res)

print("Mengekstrak fitur Spoof...")
for fid in spoof_list:
    res = extract_file_stats(fid, "Spoof")
    if res:
        all_results.append(res)

# 5. KALKULASI STATISTIK DESKRIPTIF (TABEL B)
df_final = pd.DataFrame(all_results)


def get_stats_table(df, feature_name):
    stats = (
        df.groupby("Label")[feature_name]
        .agg(["mean", "std", "min", "max", "median"])
        .reset_index()
    )

    # Tambahkan Q1, Q3, IQR, dan Skewness
    q1 = df.groupby("Label")[feature_name].quantile(0.25).values
    q3 = df.groupby("Label")[feature_name].quantile(0.75).values
    skew_val = df.groupby("Label")[feature_name].apply(skew).values

    stats["Q1"] = q1
    stats["Q3"] = q3
    stats["IQR"] = q3 - q1
    stats["Skewness"] = skew_val
    stats["Fitur"] = feature_name
    return stats


# Gabungkan semua fitur ke satu tabel akhir
features_to_report = [
    "LFCC_Mean",
    "CQT_dB_Mean",
    "RMS_Mean",
    "ZCR_Mean",
    "Centroid_Hz",
    "Rolloff_Hz",
    "F0_mean",
    "Duration",
]
table_b_rows = [get_stats_table(df_final, f) for f in features_to_report]
table_b_final = pd.concat(table_b_rows).sort_values(by=["Fitur", "Label"])

# Export ke CSV untuk Excel
table_b_final.to_csv("Tabel_B_Output.csv", index=False)
print("\nProses Selesai! Data Tabel B telah disimpan ke 'Tabel_B_Output.csv'")
print(table_b_final.head(10))
