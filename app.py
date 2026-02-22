import torch
import torchaudio
import gradio as gr
from pydub import AudioSegment
from nemo.collections.speechlm2.models import SALM
import os
import math

# 1. Load Model (RTX 4090 Optimized)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Canary-Qwen-2.5B to {device}...")
model = SALM.from_pretrained("nvidia/canary-qwen-2.5b").bfloat16().to(device)
model.eval()

def process_and_transcribe(input_path):
    if not input_path:
        return "No audio file provided."

    try:
        # 2. Load Audio & Preprocess (Mono, 16kHz, Normalized)
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        
        # Peak Normalization (helps with background music)
        audio = audio.normalize()
        
        duration_ms = len(audio)
        chunk_length_ms = 30000  # 30 seconds (staying safe under 40s limit)
        full_transcript = []
        
        num_chunks = math.ceil(duration_ms / chunk_length_ms)
        print(f"Slicing {duration_ms/1000:.1f}s audio into {num_chunks} chunks...")

        # 3. Process Chunks
        for i in range(0, duration_ms, chunk_length_ms):
            chunk = audio[i : i + chunk_length_ms]
            temp_path = f"temp_chunk_{i}.wav"
            chunk.export(temp_path, format="wav")

            prompt_text = f"Transcribe the following: {model.audio_locator_tag}"
            prompts = [[{"role": "user", "content": prompt_text, "audio": [temp_path]}]]

            with torch.no_grad():
                # Loop-prevention settings
                answer_ids = model.generate(
                    prompts=prompts, 
                    max_new_tokens=256,
                    do_sample=False, 
                    repetition_penalty=1.2
                )
            
            chunk_text = model.tokenizer.ids_to_text(answer_ids[0].cpu())
            chunk_text = chunk_text.replace("<|endoftext|>", "").strip()
            
            if chunk_text:
                full_transcript.append(chunk_text)
            
            # Cleanup temp file immediately
            os.remove(temp_path)
            print(f"Finished chunk {len(full_transcript)}/{num_chunks}")

        return " ".join(full_transcript) if full_transcript else "[No speech detected]"

    except Exception as e:
        return f"Error: {str(e)}"

# 4. Gradio UI Setup
ui = gr.Interface(
    fn=process_and_transcribe,
    inputs=gr.Audio(type="filepath", label="Upload Long Audio (7:52 file ok!)"),
    outputs=gr.Textbox(label="Full Transcription"),
    title="NVIDIA Canary-Qwen-2.5B (Long-Form Mode)",
    description="Automatically slices long audio into 30s chunks to prevent 'comma loops' and hallucinations."
)

if __name__ == "__main__":
    ui.launch()