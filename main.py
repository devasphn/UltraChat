import torch
import gc
import librosa
import tempfile
import torchaudio
import gradio as gr
from transformers import pipeline
from chatterbox.tts import ChatterboxTTS

# 1. Ultravox pipeline with pinned revision and CUDA device
uv_pipe = pipeline(
    model="fixie-ai/ultravox-v0_4",
    trust_remote_code=True,
    revision="main",
    device=0 if torch.cuda.is_available() else -1
)

# 2. Chatterbox TTS loader and synthesizer
class TTS:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def load(self):
        if self.model is None:
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
        return True

    def synthesize(self, text: str) -> str:
        self.load()
        # Generate waveform tensor or numpy array
        wav = self.model.generate(text)
        # Ensure torch.Tensor
        if not isinstance(wav, torch.Tensor):
            wav = torch.from_numpy(wav)
        # Add channel dimension: (samples,) -> (1, samples)
        wav = wav.unsqueeze(0)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        torchaudio.save(tmp.name, wav, self.model.sr)
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        return tmp.name

tts = TTS()

# 3. Speech-to-speech function
def s2s(audio_path: str) -> str:
    # Load user audio
    audio, sr = librosa.load(audio_path, sr=16000)
    # Conversation turns
    turns = [
        {"role": "system", "content": "You are a helpful voice assistant."},
        {"role": "user",   "content": "Respond to the following audio input."}
    ]
    # Ultravox generates text
    result = uv_pipe(
        {"audio": audio, "turns": turns, "sampling_rate": sr},
        max_new_tokens=128
    )
    text = result[0]["generated_text"]
    # Chatterbox converts text to speech
    return tts.synthesize(text)

# 4. Gradio web UI
with gr.Blocks(title="UltraChat S2S Agent") as demo:
    gr.Markdown("# 🎙️ UltraChat: Speech-to-Speech AI Agent")
    audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Your Speech")
    btn = gr.Button("▶️ Talk to AI")
    audio_out = gr.Audio(type="filepath", label="AI Response")
    btn.click(fn=s2s, inputs=audio_in, outputs=audio_out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
