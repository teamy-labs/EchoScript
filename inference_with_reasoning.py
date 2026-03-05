import librosa
import torch
import torch.nn.functional as F
from models.bart_captioning_v2 import BartCaptionModelV2  # Import the V2 model
import sys
import os

# --- Configuration ---
# =========================================================================
# >>> 1. SET THE PATH TO YOUR CHECKPOINT FILE HERE <<<
checkpoint_path = "C:/Users/anike/OneDrive/Desktop/my files/langs/snake/notebooks/wavcaps/htsat_bart.pt"
# =========================================================================

# --- User Input and Validation ---
audio_path = input("Enter the path to the audio file you want to caption: ")

if not os.path.exists(audio_path):
    print(f"Error: Audio file not found at {audio_path}. Please check the file path.")
    sys.exit(1)

# --- Model Loading and Setup ---
try:
    print(f"Loading model checkpoint from {checkpoint_path}...")
    cp = torch.load(checkpoint_path, map_location=torch.device("cpu"))
except FileNotFoundError:
    print(f"Error: Checkpoint file not found at {checkpoint_path}.")
    sys.exit(1)

config = cp["config"]

# Determine the device
device_str = config.get("device", "cpu")
device = torch.device(
    "cuda" if torch.cuda.is_available() and "cuda" in device_str else "cpu"
)
print(f"Using device: {device}")

# Initialize the V2 model architecture and load the checkpoint weights
model = BartCaptionModelV2(config)
model.load_state_dict(cp["model"])
model.to(device)
model.eval()

# --- Audio Loading and Preprocessing (Simplified, focusing on HTSAT 10s logic) ---
sr = config["audio_args"]["sr"]
print(f"Loading and processing audio file: {audio_path}...")
try:
    waveform, loaded_sr = librosa.load(audio_path, sr=sr, mono=True)
    if loaded_sr != sr:
        print(f"Warning: Audio resampled from {loaded_sr}Hz to {sr}Hz.")
    waveform = torch.tensor(waveform).float()
except Exception as e:
    print(f"Error loading audio file: {e}")
    sys.exit(1)

# HTSAT (transformer) encoder truncation/padding logic
max_length_sec = 10
max_length = sr * max_length_sec

if len(waveform) > max_length:
    print(f"Warning: Audio truncated to {max_length_sec} seconds.")
    waveform = waveform[:max_length]
elif len(waveform) < max_length:
    print(f"Padding audio to {max_length_sec} seconds.")
    waveform = F.pad(waveform, [0, max_length - len(waveform)], "constant", 0.0)

waveform = waveform.unsqueeze(0).to(device)  # Add batch dimension and move to device

# --- Caption Generation with Reasoning (Top 3) ---
print("\nGenerating top 3 caption sequences...")
# We use num_beams=3 and num_return_sequences=3 to get the top 3 results
top_3_results = model.generate(
    samples=waveform,
    num_beams=3,
    num_return_sequences=3,
    max_length=50,  # Ensure enough length for a full caption
)

# --- Process and Display Reasoning Output ---
if top_3_results:
    best_caption = top_3_results[0][0]
    best_score = top_3_results[0][1]

    print("\n" + "=" * 60)
    print("      A U D I O   C A P T I O N I N G   R E A S O N I N G      ")
    print("=" * 60)

    # Best Caption Output
    print(f"\nFINAL CAPTION (RANK 1, BEST SCORE):")
    print(f'  -> "{best_caption}"')

    # Reasoning Output
    print(
        f"\nREASONING: The model selected this caption because it achieved the highest cumulative log-probability score during beam search. The top 3 alternative sequences considered were:"
    )
    print("-" * 60)

    for rank, (caption, score) in enumerate(top_3_results):
        # Calculate a descriptive confidence measure (relative to the best score)
        # Higher score (less negative) is better.
        confidence_delta = (score - best_score) / max_length_sec

        print(f"**Rank {rank + 1}:**")
        print(f'  - **Caption:** "{caption}"')
        print(f"  - **Log-Probability Score:** {score:.4f} (Used for ranking)")

        # The justification and status lines have been removed as requested.

        print("-" * 60)
else:
    print("\nError: Caption generation failed or returned no results.")
