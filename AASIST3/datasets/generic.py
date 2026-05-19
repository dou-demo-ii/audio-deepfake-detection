import os
import pandas as pd
import torchaudio
from pathlib import Path
from typing import Optional, Callable, Dict, List
from tqdm import tqdm
from torch.utils.data import Dataset
import soundfile as sf
import torch

from .utils import apply_random_segment_extraction, print_fancy


class BaseAudioDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        annot: Optional[str] = None,
        sample_rate: int = 16000,
        max_length: int = 64600,
        return_file_path: bool = False
    ):
        if annot and os.path.exists(annot):
            metadata = pd.read_csv(annot)
        self.metadata = metadata.reset_index(drop=True)
        self.sr = sample_rate
        self.max_length = max_length
        self.return_file_path = return_file_path

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]

        if str(row['label']).isnumeric(): lbl = int(row["label"])
        else: lbl = 0 if row['label'] == "spoof" else 1

        path = row["audio_path"]
        try:
            audio_data, sr = sf.read(path)
            audio = torch.from_numpy(audio_data).float()
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            else:
                audio = audio.T
        except RuntimeError as er:
            print_fancy(f"Error while trying to open file {path}: {er}", style="error")
            return self.__getitem__(idx + 1)

        if audio.size(0) == 2:
            audio = audio.mean(0, keepdim=True)
        if sr != self.sr:
            audio = torchaudio.functional.resample(audio, sr, self.sr)
        audio = torchaudio.functional.preemphasis(audio)
        audio = apply_random_segment_extraction(audio, self.max_length)

        if not self.return_file_path: return audio, lbl
        else: return audio, lbl, path


class TrialAudioDataset(BaseAudioDataset):
    def __init__(
        self,
        root_dir: str,
        trial_file: str,
        first_idx: int,
        label_idx: int,
        extension: str,
        **kwargs
    ):
        df = self.build_trial_df(
            root_dir, trial_file,
            first_idx, label_idx,
            extension=extension,
        )
        super().__init__(metadata=df, **kwargs)

    def build_trial_df(
            self,
            root_dir: str,
            trial_file: str,
            first_idx: int,
            label_idx: int,
            extension: str = ".flac",
            label_map: Callable[[str], int] = lambda x: 0 if x == "spoof" else 1,

    ) -> pd.DataFrame:
        lines = Path(trial_file).read_text().splitlines()
        if not lines:
            raise RuntimeError(f"Empty trial file: {trial_file}")
        rows = [ln.split() for ln in lines if ln.strip()]
        paths = [os.path.join(root_dir, row[first_idx] + extension) for row in rows]
        labels = [label_map(row[label_idx]) for row in rows]
        df = pd.DataFrame({"audio_path": paths, "label": labels})
        return df


def discover_files_recursively(
    directory_root: str,
    extension_filter: Optional[str] = None
) -> List[str]:
    discovered_paths = []
    search_root = Path(directory_root)

    if not search_root.exists() or not search_root.is_dir():
        return discovered_paths

    file_iterator = search_root.rglob("*")

    for current_path in tqdm(file_iterator, desc="Scanning files"):
        if current_path.is_file():
            if extension_filter is None:
                discovered_paths.append(str(current_path))
            elif current_path.suffix.lower() == extension_filter.lower():
                discovered_paths.append(str(current_path))

    return discovered_paths


def build_list_df(
    root_dir: str,
    extension: str = ".wav",
    label_map: Optional[Dict[str, int]] = None,
    save_annot: bool = False,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    files = discover_files_recursively(root_dir, extension)
    if label_map is None:
        labels = [0] * len(files)
    elif isinstance(label_map, dict):
        labels = [label_map.get(Path(fp).parent.name, 0) for fp in files]
    else:
        raise ValueError("label_map must be a dict or None")
    df = pd.DataFrame({"audio_path": files, "label": labels})
    if save_annot and save_path:
        df.to_csv(save_path, index=False)
    return df


class FolderAudioDataset(BaseAudioDataset):
    def __init__(
        self,
        root_dir: str,
        extension: str,
        label_map: Optional[Dict[str, int]],
        **kwargs
    ):
        df = build_list_df(
            root_dir, extension=extension,
            label_map=label_map,
        )
        super().__init__(metadata=df, **kwargs)