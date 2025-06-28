import torch, gc, librosa, tempfile, gradio as gr
from transformers import pipeline
from chatterbox.tts import ChatterboxTTS

# 1. Ultravox Pipeline
uv_pipe = pipeline(
    task="audio-to-text",
    model="fixie-ai/ultravox-v0_4",
    trust_remote_code=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 2. Chatterbox TTS Loader
class TTS:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
    def load(self):
        if not self.model:
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
        return True
    def synthesize(self, text):
        self.load()
        wav = self.model.generate(text)
        temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        torch.save(wav, temp.name)
        if self.device=="cuda":
            torch.cuda.empty_cache(); gc.collect()
        return temp.name

tts = TTS()

# 3. Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🎤 Speech-to-Speech Agent")
    audio_in = gr.Audio(source="microphone", type="filepath", label="Your Speech")
    btn = gr.Button("▶️ Generate Response")
    audio_out = gr.Audio(label="AI Reply")
    def s2s(audio_path):
        # ASR + LLM via Ultravox
        audio, sr = librosa.load(audio_path, sr=16000)
        result = uv_pipe({
            "audio": audio,
            "turns": [{"role":"system","content":"You are a helpful assistant."}],
            "sampling_rate": sr
        }, max_new_tokens=128)
        response_text = result[0]["generated_text"]
        # TTS
        wav_path = tts.synthesize(response_text)
        return wav_path
    btn.click(s2s, inputs=audio_in, outputs=audio_out)

if __name__=="__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
