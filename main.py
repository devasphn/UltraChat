import torch
import gc
import librosa
import tempfile
import gradio as gr
from transformers import pipeline
from chatterbox.tts import ChatterboxTTS

# 1. Ultravox Pipeline (auto-detect, pinned to 'main' revision)
uv_pipe = pipeline(
    model="fixie-ai/ultravox-v0_4",
    trust_remote_code=True,
    revision="main"  # Pin to a known-safe revision
)

# 2. Chatterbox TTS Loader
class TTS:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def load(self):
        if self.model is None:
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
        return True

    def synthesize(self, text):
        self.load()
        wav = self.model.generate(text)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        torch.save(wav, tmp.name)
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        return tmp.name

tts = TTS()

# 3. Speech-to-Speech Function
def s2s(audio_path):
    # Load input audio
    audio, sr = librosa.load(audio_path, sr=16000)
    # Build system prompt & user turns
    turns = [
        {"role": "system", "content": "You are a helpful voice assistant."},
        {"role": "user",   "content": "Respond to the following audio input."}
    ]
    # Ultravox generates text from audio
    result = uv_pipe(
        {"audio": audio, "turns": turns, "sampling_rate": sr},
        max_new_tokens=128
    )
    response_text = result[0]["generated_text"]
    # Chatterbox TTS synthesizes speech
    return tts.synthesize(response_text)

# 4. Gradio Web UI
with gr.Blocks(title="UltraChat S2S Agent") as demo:
    gr.Markdown("# 🎙️ UltraChat: Speech-to-Speech AI Agent")
    audio_in = gr.Audio(source="microphone", type="filepath", label="Your Speech")
    btn = gr.Button("▶️ Talk to AI")
    audio_out = gr.Audio(label="AI Response")
    btn.click(fn=s2s, inputs=audio_in, outputs=audio_out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
