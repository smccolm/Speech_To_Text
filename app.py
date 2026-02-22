import torch, torchaudio, gradio as gr, os, re
from pydub import AudioSegment, silence
from nemo.collections.speechlm2.models import SALM
import Levenshtein

# Load Model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Canary-Qwen-2.5B to {device}...")
model = SALM.from_pretrained("nvidia/canary-qwen-2.5b").bfloat16().to(device)
model.eval()

def clean_for_match(text):
    """Normalize text for matching by removing punctuation and casing."""
    return re.sub(r'[^a-zA-Z0-9\s]', '', text).lower().strip()

def merge_transcripts(text1, text2):
    """Advanced OLC Stitching with punctuation-blind matching and 40-word window."""
    words1 = text1.split()
    words2 = text2.split()
    if not words1: return text2
    if not words2: return text1

    # Increased window to 40 to catch full-sentence duplicates
    search_range = min(len(words1), len(words2), 40)
    best_overlap = 0

    # Iterate backwards to find the LONGEST match first (Greedy Alignment)
    for i in range(search_range, 0, -1):
        suffix = " ".join(words1[-i:])
        prefix = " ".join(words2[:i])
        
        # Match against cleaned versions to ignore Punctuation/Case drift
        if Levenshtein.ratio(clean_for_match(suffix), clean_for_match(prefix)) > 0.85:
            best_overlap = i
            break 

    return " ".join(words1 + words2[best_overlap:])

def process_and_transcribe(input_path):
    if not input_path: return "No audio provided.", None
    try:
        audio = AudioSegment.from_file(input_path).set_frame_rate(16000).set_channels(1).normalize()
        duration_ms = len(audio)
        full_transcript = ""
        current_pos = 0
        
        # Target 30s chunks, but find silence to avoid mid-word cuts ("costs temp")
        while current_pos < duration_ms:
            target_end = current_pos + 30000
            if target_end >= duration_ms:
                end_pos = duration_ms
            else:
                # Search for silence in a 2-second window around the target
                silence_search = audio[target_end-1000 : target_end+1000]
                pauses = silence.detect_silence(silence_search, min_silence_len=300, silence_thresh=-40)
                if pauses:
                    # Cut at the middle of the first detected pause
                    pause_mid = (pauses[0][0] + pauses[0][1]) // 2
                    end_pos = (target_end - 1000) + pause_mid
                else:
                    end_pos = target_end # Fallback to blunt cut

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
            # Move forward but maintain a 5s "Safety Buffer" for the OLC Merger
            current_pos = max(0, end_pos - 5000)

        file_path = "transcript.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_transcript)
            
        return full_transcript, file_path
    except Exception as e:
        return f"Error: {str(e)}", None

ui = gr.Interface(
    fn=process_and_transcribe,
    inputs=gr.Audio(type="filepath", label="Upload Audio"),
    outputs=[gr.Textbox(label="Clean Transcript"), gr.File(label="Download TXT")],
    title="Canary 2.5B: Overlap-Layout-Consensus (OLC) stitching",
    description="DNA-Assembly v2: Silence-aware slicing + Punctuation-blind stitching.",
    flagging_mode="never"
)

if __name__ == "__main__": ui.launch()