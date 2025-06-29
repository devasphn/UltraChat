import torch
import gc
import librosa
import tempfile
import torchaudio
import gradio as gr
from transformers import pipeline # Use pipeline for Ultravox
from chatterbox.tts import ChatterboxTTS # Ensure this import is here

# 1. Ultravox pipeline - FIXED
try:
    print(f"Using device: {'cuda:0' if torch.cuda.is_available() else 'cpu'}")

    # Load the Ultravox model using the pipeline - NO TASK SPECIFIED
    # This is the correct way according to the official documentation
    uv_pipe = pipeline(
        model="fixie-ai/ultravox-v0_4",
        trust_remote_code=True,
        device_map="auto", # Let pipeline handle device placement
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    print("Ultravox pipeline loaded successfully.")

except Exception as e:
    print(f"Error loading Ultravox pipeline: {e}")
    exit()

# 2. Chatterbox TTS loader (no changes needed here, it works)
class TTS:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def load(self):
        if self.model is None:
            print(f"Loading Chatterbox TTS on device: {self.device}")
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            print("Chatterbox TTS loaded successfully.")
        return True

    def synthesize(self, text: str) -> str:
        self.load()
        if not text.strip(): # Handle empty text
            print("Warning: Received empty text for TTS, returning empty audio.")
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            torchaudio.save(tmp.name, torch.zeros(1, 16000), 16000)
            return tmp.name

        wav = self.model.generate(text)
        if not isinstance(wav, torch.Tensor):
            wav = torch.from_numpy(wav)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        torchaudio.save(tmp.name, wav.cpu(), self.model.sr)
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        return tmp.name

tts = TTS()

# 3. Speech-to-speech function - UPDATED FOR CORRECT API
def s2s(audio_path: str) -> str:
    if audio_path is None:
        print("No audio input received yet. Waiting for user interaction.")
        return None

    print(f"Processing audio from: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000)
    print(f"Audio loaded with sample rate: {sr}")

    # The correct format according to Ultravox documentation
    turns = [
        {
            "role": "system",
            "content": "You are a friendly and helpful voice assistant. Respond naturally and conversationally."
        }
    ]
    print("Conversation turns prepared for pipeline.")

    # Pass audio, turns, and sampling_rate to the pipeline
    # This is the correct API according to the official documentation
    result = uv_pipe({
        'audio': audio, 
        'turns': turns, 
        'sampling_rate': sr
    }, max_new_tokens=128)
    
    # Extract the response text from the result
    response_text = result
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], dict) and "generated_text" in result[0]:
            response_text = result[0]["generated_text"]
        else:
            response_text = str(result[0])
    elif isinstance(result, dict) and "generated_text" in result:
        response_text = result["generated_text"]
    else:
        response_text = str(result)
    
    print(f"Generated text response from pipeline: {response_text}")

    # Clean up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("GPU cache cleared.")

    # Convert text → speech via Chatterbox
    print("Synthesizing speech with Chatterbox TTS...")
    synthesized_audio_path = tts.synthesize(response_text)
    print(f"Speech synthesis complete. Output audio: {synthesized_audio_path}")
    return synthesized_audio_path

# 4. Gradio Web UI
with gr.Blocks(title="UltraChat S2S Agent") as demo:
    gr.Markdown("# 🎙️ UltraChat: Speech-to-Speech AI Agent")
    audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Your Speech")
    btn = gr.Button("▶️ Talk to AI")
    audio_out = gr.Audio(type="filepath", label="AI Response")
    btn.click(fn=s2s, inputs=audio_in, outputs=audio_out)

if __name__ == "__main__":
    print("Launching Gradio demo...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
