from .generic import TrialAudioDataset


class ASVspoof2019Train(TrialAudioDataset):
    def __init__(self, root_dir, meta_path, subset=None, save_annot=False, **kwargs):
        super().__init__(
            root_dir, meta_path,
            first_idx=1, label_idx=4,
            extension=".flac",
            **kwargs
        )


class ASVspoof5Train(TrialAudioDataset):
    def __init__(self, root_dir, meta_path, subset=None, save_annot=False, **kwargs):
        super().__init__(
            root_dir, meta_path,
            first_idx=1, label_idx=5,
            extension=".flac",
            **kwargs
        )


class ASVspoof2021LA(TrialAudioDataset):
    def __init__(self, root_dir, meta_path, subset=None, save_annot=False, **kwargs):
        super().__init__(
            root_dir, meta_path,
            first_idx=1, label_idx=5,
            extension=".flac",
            **kwargs
        )


class ASVspoof2021DF(TrialAudioDataset):
    def __init__(self, root_dir, meta_path, subset=None, save_annot=False, **kwargs):
        super().__init__(
            root_dir, meta_path,
            first_idx=1, label_idx=5,
            extension=".flac",
            **kwargs
        )


class ASVspoof5Dev(TrialAudioDataset):
    def __init__(self, root_dir, meta_path, subset=None, save_annot=False, **kwargs):
        super().__init__(
            root_dir, meta_path,
            first_idx=1, label_idx=2,
            extension=".flac",
            **kwargs
        )


class ASVspoof2019Dev(TrialAudioDataset):
    def __init__(self, root_dir, meta_path, subset=None, save_annot=False, **kwargs):
        super().__init__(
            root_dir, meta_path,
            first_idx=1, label_idx=4,
            extension=".flac",
            **kwargs
        )


class ASVspoof5Eval(TrialAudioDataset):
    def __init__(self, root_dir, meta_path, subset=None, save_annot=False, **kwargs):
        super().__init__(
            root_dir, meta_path,
            first_idx=1, label_idx=8,
            extension=".flac",
            **kwargs
        )


class ASVspoof2019Eval(TrialAudioDataset):
    def __init__(self, root_dir, meta_path, subset=None, save_annot=False, **kwargs):
        super().__init__(
            root_dir, meta_path,
            first_idx=1, label_idx=4,
            extension=".flac",
            **kwargs
        )
