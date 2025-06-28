import torch
import gc
import librosa
import tempfile
import torchaudio
import gradio as gr
from transformers import pipeline
from chatterbox.tts import ChatterboxTTS

# 1. Ultravox pipeline
#    - Remove device_map/offload to avoid meta‐tensor errors
#    - Load full model on CPU then move to GPU via `device=0`
#    - Use float16 for reduced VRAM footprint
uv_pipe = pipeline(
    task="audio-text-to-text",
    model="fixie-ai/ultravox-v0_4",
    trust_remote_code=True,
    revision="main",  
    device=0,                          # send model to GPU 0 after loading  
    torch_dtype=torch.float16         # load weights in half precision
)

# 2. Chatterbox TTS loader
class TTS:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def load(self):
        if self.model is None:
            # Load Chatterbox on the same device
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
        return True

    def synthesize(self, text: str) -> str:
        self.load()
        # Generate waveform (Tensor or NumPy)
        wav = self.model.generate(text)
        if not isinstance(wav, torch.Tensor):
            wav = torch.from_numpy(wav)
        # Ensure shape (1, samples) for torchaudio
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        # Save to temporary file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        torchaudio.save(tmp.name, wav, self.model.sr)
        # Clear GPU cache
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        return tmp.name

tts = TTS()

# 3. Speech-to-speech function
def s2s(audio_path: str) -> str:
    # Load user audio at 16 kHz
    audio, sr = librosa.load(audio_path, sr=16000)
    # Prepare system + user turns
    turns = [
        {"role": "system", "content": "You are a helpful voice assistant."},
        {"role": "user",   "content": "Please respond to my speech input."}
    ]
    # Convert speech → text via Ultravox
    result = uv_pipe(
        {"audio": audio, "turns": turns, "sampling_rate": sr},
        max_new_tokens=128
    )
    response_text = result[0]["generated_text"]
    # Convert text → speech via Chatterbox
    return tts.synthesize(response_text)

# 4. Gradio Web UI
with gr.Blocks(title="UltraChat S2S Agent") as demo:
    gr.Markdown("# 🎙️ UltraChat: Speech-to-Speech AI Agent")
    audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Your Speech")
    btn = gr.Button("▶️ Talk to AI")
    audio_out = gr.Audio(type="filepath", label="AI Response")
    btn.click(fn=s2s, inputs=audio_in, outputs=audio_out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
