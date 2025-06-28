import torch
import gc
import librosa
import tempfile
import torchaudio
import gradio as gr
from transformers import AutoProcessor, AutoModel
from chatterbox.tts import ChatterboxTTS # Make sure this is present

# 1. Ultravox model and processor loading
try:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = AutoProcessor.from_pretrained(
        "fixie-ai/ultravox-v0_4",
        trust_remote_code=True,
        revision="main"
    )
    print("Ultravox processor loaded successfully.")

    model = AutoModel.from_pretrained(
        "fixie-ai/ultravox-v0_4",
        trust_remote_code=True,
        revision="main",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    print("Ultravox model loaded successfully with device_map='auto'.")

except Exception as e:
    print(f"Error loading Ultravox model: {e}")
    print("This error usually indicates a memory issue (not enough VRAM/RAM)")
    print("or an incompatibility with how the model's custom code interacts with Transformers/PyTorch's loading mechanism.")
    print("1. Check your GPU memory usage (`nvidia-smi`).")
    print("2. Ensure `accelerate` is installed and up-to-date (`pip install --upgrade accelerate`).")
    print("3. Consider if your environment has enough RAM for initial loading.")
    exit()

# 2. Chatterbox TTS loader
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
        # Handle cases where text might be very short or problematic
        if not text.strip():
            print("Warning: Received empty text for TTS, returning empty audio.")
            # Create a silent audio file or return None if that's acceptable for Gradio
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            torchaudio.save(tmp.name, torch.zeros(1, 16000), 16000) # 1 second of silence
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

# 3. Speech-to-speech function
def s2s(audio_path: str) -> str:
    if audio_path is None:
        print("No audio input received yet. Waiting for user interaction.")
        return None

    print(f"Processing audio from: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000)
    print(f"Audio loaded with sample rate: {sr}")

    # *** CRUCIAL CHANGE: Include the <|audio|> pseudo-token ***
    # The model documentation states it expects this token.
    # The processor should replace it with audio embeddings.
    turns = [
        {"role": "system", "content": "You are a helpful voice assistant. You will respond to speech inputs."},
        {"role": "user", "content": "<|audio|>"} # The audio input is implicitly linked here
    ]
    print("Conversation turns prepared with <|audio|> token.")

    # Process audio and generate text using the loaded model
    inputs = processor(
        audio=audio,
        sampling_rate=sr,
        turns=turns,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print("Inputs processed and moved to model device.")

    # Generate the output from the model
    with torch.no_grad():
        print("Generating response from Ultravox model...")
        
        generate_kwargs = {"max_new_tokens": 128}
        
        # Pass attention_mask if the processor provides it.
        # The warning "attention mask is not set" is often related to the tokenizer
        # for the *text* part of the model. By correctly structuring the input
        # with <|audio|> and letting the processor handle it, this warning
        # *might* resolve itself or become benign if the model's custom code
        # correctly manages attention internally after the audio encoding.
        if "attention_mask" in inputs:
            generate_kwargs["attention_mask"] = inputs["attention_mask"]

        # Also, explicitly set pad_token_id and eos_token_id in generate if they are the same
        # This often suppresses the warning without changing behavior if the model is robust
        if hasattr(processor, 'tokenizer') and processor.tokenizer.pad_token_id is not None and processor.tokenizer.eos_token_id is not None:
             if processor.tokenizer.pad_token_id == processor.tokenizer.eos_token_id:
                 generate_kwargs["pad_token_id"] = processor.tokenizer.eos_token_id
                 # The warning is about *inference* behavior. If pad_token_id == eos_token_id,
                 # it can make the model generate infinitely or stop prematurely.
                 # Setting it explicitly might help, or it might just suppress the warning.

        generated_ids = model.generate(**inputs, **generate_kwargs)

    print("Ultravox model generation complete.")

    # Decode the generated text
    response_text = processor.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated text response: {response_text}")

    # Clean up memory
    del inputs
    del generated_ids
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
