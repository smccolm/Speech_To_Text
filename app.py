import torch, torchaudio, gradio as gr, os, math
from pydub import AudioSegment
from nemo.collections.speechlm2.models import SALM
import Levenshtein

# Load Model (RTX 4090 Optimized)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Canary-Qwen-2.5B to {device}...")
model = SALM.from_pretrained("nvidia/canary-qwen-2.5b").bfloat16().to(device)
model.eval()

def merge_transcripts(text1, text2):
    """OLC Stitching logic to prevent word doubling at junctions."""
    words1 = text1.split()
    words2 = text2.split()
    if not words1: return text2
    if not words2: return text1
    best_overlap = 0
    search_range = min(len(words1), len(words2), 15) 
    for i in range(1, search_range + 1):
        suffix = " ".join(words1[-i:])
        prefix = " ".join(words2[:i])
        if Levenshtein.ratio(suffix.lower(), prefix.lower()) > 0.8:
            best_overlap = i
    return " ".join(words1 + words2[best_overlap:])

def process_and_transcribe(input_path):
    if not input_path: return "No audio provided.", None
    try:
        audio = AudioSegment.from_file(input_path).set_frame_rate(16000).set_channels(1).normalize()
        duration_ms, chunk_ms, overlap_ms = len(audio), 30000, 5000
        full_transcript, current_pos = "", 0
        
        while current_pos < duration_ms:
            end_pos = min(current_pos + chunk_ms, duration_ms)
            chunk = audio[current_pos:end_pos]
            temp_path = f"temp_{current_pos}.wav"
            chunk.export(temp_path, format="wav")
            
            prompts = [[{"role": "user", "content": f"Transcribe: {model.audio_locator_tag}", "audio": [temp_path]}]]
            with torch.no_grad():
                ids = model.generate(prompts=prompts, max_new_tokens=256, do_sample=False, repetition_penalty=1.2)
            
            new_text = model.tokenizer.ids_to_text(ids[0].cpu()).replace("<|endoftext|>", "").strip()
            full_transcript = merge_transcripts(full_transcript, new_text)
            
            os.remove(temp_path)
            if end_pos == duration_ms: break
            current_pos += (chunk_ms - overlap_ms)

        # Create download file
        file_path = "transcript.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_transcript)
            
        return full_transcript, file_path
    except Exception as e:
        return f"Error: {str(e)}", None

# Updated Gradio Interface for 4.x/5.x Compatibility
ui = gr.Interface(
    fn=process_and_transcribe,
    inputs=gr.Audio(type="filepath", label="Upload Audio"),
    outputs=[
        gr.Textbox(label="Live Transcription Result"),
        gr.File(label="Download Transcript (.txt)")
    ],
    title="Canary 2.5B: DNA Assembly Mode",
    description="Optimized for RTX 4090. Slices long-form audio into 30s segments with 5s homology.",
    flagging_mode="never"  # Corrected keyword for Gradio 4/5
)

if __name__ == "__main__":
    ui.launch()