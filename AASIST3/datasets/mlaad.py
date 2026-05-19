from .generic import FolderAudioDataset, BaseAudioDataset, build_list_df


class MLAAD(FolderAudioDataset):
    def __init__(self, root_dir, meta_path=None, subset=None, save_annot=False, **kwargs):
        super().__init__(
            root_dir, extension=".wav",
            label_map=None,
            **kwargs
        )


class MAILABS(BaseAudioDataset):
    def __init__(self, root_dir, meta_path=None, subset=None, save_annot=False, **kwargs):
        df = build_list_df(
            root_dir, extension=".wav",
            label_map={"bonafide": 1},
        )
        super().__init__(metadata=df, **kwargs)