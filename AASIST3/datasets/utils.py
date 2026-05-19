import torch
import random
import datetime
from typing import Literal



def apply_random_segment_extraction(
    input_tensor: torch.Tensor, 
    target_length: int = 64600
) -> torch.Tensor:
    processed_tensor = input_tensor.clone()
    while processed_tensor.ndim > 1:
        processed_tensor = processed_tensor.squeeze(0)
    
    current_length = processed_tensor.shape[0]

    if current_length > target_length:
        start_position = random.randint(0, current_length - target_length)
        return processed_tensor[start_position:start_position + target_length]

    repetition_factor = (target_length // current_length) + 1
    extended_tensor = processed_tensor.repeat(repetition_factor)
    
    return extended_tensor[:target_length]


def print_fancy(
    message: str,
    style: Literal["info", "success", "warning", "error", "step", "header"] = "info",
    width: int = 80,
    add_timestamp: bool = True,
    emoji: bool = True
) -> None:
    colors = {
        "info": "\033[94m",      # Blue
        "success": "\033[92m",   # Green
        "warning": "\033[93m",   # Yellow
        "error": "\033[91m",     # Red
        "step": "\033[96m",      # Cyan
        "header": "\033[95m",    # Magenta
        "reset": "\033[0m",      # Reset
        "bold": "\033[1m",       # Bold
        "dim": "\033[2m"         # Dim
    }

    emojis = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "step": "🔄",
        "header": "🎯"
    } if emoji else {key: "" for key in ["info", "success", "warning", "error", "step", "header"]}
    
    timestamp = datetime.datetime.now().strftime("%H:%M:%S") if add_timestamp else ""

    parts = []
    if emoji and style in emojis:
        parts.append(emojis[style])
    if timestamp:
        parts.append(f"[{timestamp}]")
    parts.append(message)
    
    full_message = " ".join(parts)
    
    color = colors.get(style, colors["info"])
    bold = colors["bold"] if style in ["header", "error"] else ""
    reset = colors["reset"]

    border = "=" * width
    
    print()
    print(f"{color}{bold}{border}{reset}")
    print(f"{color}{full_message}{reset}")
    print(f"{color}{bold}{border}{reset}")
    print()