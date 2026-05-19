import torch, torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config


class Wav2Vec2Encoder(nn.Module):
    """SSL encoder based on Hugging Face's Wav2Vec2 model."""

    def __init__(self,
                 model_name_or_path: str = "facebook/wav2vec2-large-xlsr-53",
                 ssl_out_dim: int = 768,
                 use_ssl_n_layers: int = None,
                 freeze_ssl_n_layers: int = 0,
                 output_attentions: bool = False,
                 output_hidden_states: bool = False,
                 normalize_waveform: bool = True,
                 cache_dir: str = "weights",
                 load_pretrained: bool = True):
        """Initialize the Wav2Vec2 encoder.

        Args:
            model_name_or_path: HuggingFace model name or path to local model.
            ssl_out_dim: Output dimension of the Wav2Vec2 encoder.
            use_ssl_n_layers: Number of Wav2Vec2 layers to use. If None, use all layers.
            freeze_ssl_n_layers: Number of Wav2Vec2 layers to freeze during training.
            output_attentions: Whether to output attentions.
            output_hidden_states: Whether to output hidden states.
            normalize_waveform: Whether to normalize the waveform input.
            cache_dir: Directory to cache pretrained models.
            load_pretrained: Whether to load pretrained weights. If False, initializes with random weights.
        """
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.ssl_out_dim = ssl_out_dim
        self.use_ssl_n_layers = use_ssl_n_layers
        self.freeze_ssl_n_layers = freeze_ssl_n_layers
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.normalize_waveform = normalize_waveform

        if load_pretrained:
            self.model = Wav2Vec2Model.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        else:
            config = Wav2Vec2Config.from_pretrained(
                model_name_or_path, 
                cache_dir=cache_dir,
                local_files_only=False
            )
            self.model = Wav2Vec2Model(config)
            self.model.init_weights()

    def forward(self, x):
        """Forward pass through the Wav2Vec2 encoder.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels)

        Returns:
            Extracted features of shape (batch_size, sequence_length, ssl_out_dim)
        """
        # Handle shape: convert (batch_size, sequence_length, channels) to (batch_size, sequence_length)
        if x.ndim == 3:
            x = x.squeeze(-1)  # Remove channel dimension if present

        if self.normalize_waveform:
            x = x / (torch.max(torch.abs(x), dim=1, keepdim=True)[0] + 1e-8)

        outputs = self.model(
            x,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
            return_dict=True
        )

        last_hidden_state = outputs.last_hidden_state

        if self.use_ssl_n_layers is not None and self.output_hidden_states and outputs.hidden_states is not None:
            selected = outputs.hidden_states[-self.use_ssl_n_layers:]
            last_hidden_state = torch.mean(torch.stack(selected, dim=0), dim=0)
        del outputs

        return last_hidden_state