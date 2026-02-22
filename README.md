Canary-Qwen-2.5B Long-Form ASR for Windows 11

SOTA Speech-to-Text with "DNA Assembly" Audio Stitching
This repository provides a production-ready implementation of NVIDIA’s Canary-Qwen-2.5B—the top-ranked model on the Hugging Face Open ASR Leaderboard—specifically optimized for Windows 11 and NVIDIA RTX 40-series GPUs (Tested on RTX 4090).
Unlike standard implementations, this version includes a custom Overlapping-Sliding-Window ("DNA Assembly") algorithm. This allows for transcribing audio files of indefinite length (tested on 8+ minute samples) while bypassing the model's native 40-second architectural limit and preventing the "comma-hallucination" loops common in LLM-based ASR.
💻 Hardware & Software Requirements
Hardware
GPU: NVIDIA RTX 3090 / 4090 (24GB VRAM recommended).
VRAM Usage: ~8.5 GB in bfloat16 precision.
Disk Space: ~15 GB for model weights and environments.
Software
OS: Windows 11 (Version 10.0.22621+).
Python: 3.11 (Mandatory for dependency compatibility).
Package Manager: Miniconda or Anaconda (Required to resolve the pynini Windows build issue).
System Tools:
FFmpeg (Must be added to Windows System PATH).
Visual Studio Build Tools 2022 (Install the "Desktop development with C++" workload).
🛠️ Installation Step-by-Step
1. Environment Setup
Open a standard Windows CMD (not PowerShell) and run the following sequence to create the environment and solve the pynini C++ compilation barrier on Windows.
code
Cmd
conda create -n canary python=3.11 -y
conda activate canary
conda install -c conda-forge pynini=2.1.6 -y
2. Install Core Dependencies
Install the specific CUDA 12.4 wheels for PyTorch and the required audio/stitching libraries.
code
Cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install gradio Cython lhotse pydub python-Levenshtein soundfile
3. Install NVIDIA NeMo (From Source)
Canary-Qwen-2.5B requires the latest speechlm2 module, which is only available in the NeMo trunk.
code
Cmd
pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git"
🧬 The "DNA Assembly" Logic
Standard ASR slicing (Blunt Cuts) fails if a word is cut in half (e.g., "trans-" at the end of slice 1 and "-cribed" at the start of slice 2). This results in hallucinations like "transport" and "described."
Our Implementation:
Redundant Slicing: The audio is sliced into 30-second "reads" with a 5-second overlap (homology region).
Deterministic Decoding: We use do_sample=False and repetition_penalty=1.2 to ensure the model doesn't enter a loop when encountering background music.
Levenshtein Stitching: The system compares the last 15 words of "Slice A" with the first 15 words of "Slice B." It calculates the edit distance (Levenshtein Ratio) to find the exact mathematical junction where the sentences overlap and zips them together, ensuring no words are doubled or cut.
🚀 Usage
1. Starting the Application
You can launch the Gradio UI using the provided batch script:
code
Batch
:: Save this as start_app.bat
cd /d %~dp0
call conda activate canary
python app.py
pause
2. UI Parameters
Upload Audio: Supports .wav, .mp3, .flac.
Auto-Preprocessing: The script automatically converts audio to 16kHz Mono and applies Peak Normalization to ensure the speech encoder receives a consistent signal above background noise (like classical music).
📂 Project Structure
code
Text
STT/
├── app.py              # Main Gradio application with DNA Assembly logic
├── normalized_audio.wav # Temporary file for peak-normalized audio
├── temp_*.wav          # Temporary chunk files (auto-deleted)
└── start_app.bat       # Windows launcher script
⚠️ Troubleshooting
Error	Cause	Solution
FileNotFoundError: [Errno 2] ffmpeg	FFmpeg not in PATH	Install FFmpeg and add C:\ffmpeg\bin to System Environment Variables.
RuntimeError: Size mismatch	Stereo Audio	The app.py script now handles this automatically by downmixing to Mono.
,,,,,,,,,,,,,,, (Comma Loops)	Hallucination	Caused by do_sample=True or audio > 40s. Use the provided DNA Assembly script to stay under 40s.
Failed to build pynini	Pip compilation error	You must use conda install -c conda-forge pynini on Windows.
📜 Citations & Acknowledgments
NVIDIA Canary-Qwen-2.5B: Hugging Face Model Card
NeMo Framework: NVIDIA, GitHub Repository
Assembly Logic: Inspired by the Overlap-Layout-Consensus algorithms used in genomic sequencing.
Created for local inference on Windows 11 + RTX 4090.