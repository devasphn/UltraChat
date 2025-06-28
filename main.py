import torch
import gc
import librosa
import tempfile
import torchaudio
import gradio as gr
from transformers import AutoProcessor, AutoModelForCausalLM # Corrected import here
from chatterbox.tts import ChatterboxTTS

# 1. Ultravox model and processor loading
#    - Removed 'task' from pipeline as we're loading directly
#    - Use AutoProcessor and AutoModelForCausalLM for models based on Llama/causal LMs
#    - Ensure model is moved to GPU if available and float16 for efficiency
try:
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the processor and model
    processor = AutoProcessor.from_pretrained(
        "fixie-ai/ultravox-v0_4",
        trust_remote_code=True,
        revision="main"
    )
    model = AutoModelForCausalLM.from_pretrained( # Changed to AutoModelForCausalLM
        "fixie-ai/ultravox-v0_4",
        trust_remote_code=True,
        revision="main",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, # Use float16 on GPU, float32 on CPU
    ).to(device) # Move model to device after loading
    print("Ultravox model loaded successfully.")

except Exception as e:
    print(f"Error loading Ultravox model: {e}")
    print("Please ensure you have transformers and accelerate installed, and your system has enough memory.")
    print("If the error persists, check the exact class type required by 'fixie-ai/ultravox-v0_4' in its Hugging Face model code.")
    exit() # Exit if the model cannot be loaded

# 2. Chatterbox TTS loader
class TTS:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def load(self):
        if self.model is None:
            # Load Chatterbox on the same device
            print(f"Loading Chatterbox TTS on device: {self.device}")
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            print("Chatterbox TTS loaded successfully.")
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
        torchaudio.save(tmp.name, wav.cpu(), self.model.sr) # Move to CPU before saving to avoid device mismatch issues with torchaudio
        # Clear GPU cache
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        return tmp.name

tts = TTS()

# 3. Speech-to-speech function
def s2s(audio_path: str) -> str:
    print(f"Processing audio from: {audio_path}")
    # Load user audio at 16 kHz
    audio, sr = librosa.load(audio_path, sr=16000)
    print(f"Audio loaded with sample rate: {sr}")

    # Prepare system + user turns for Ultravox
    turns = [
        {"role": "system", "content": "You are a helpful voice assistant."},
        {"role": "user", "content": "Please respond to my speech input."}
    ]
    print("Conversation turns prepared.")

    # Process audio and generate text using the loaded model
    inputs = processor(
        audio=audio,
        sampling_rate=sr,
        turns=turns,
        return_tensors="pt"
    ).to(device) # Move inputs to the same device as the model
    print("Inputs processed and moved to device.")

    # Generate the output from the model
    with torch.no_grad(): # Disable gradient calculation for inference
        print("Generating response from Ultravox model...")
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    print("Ultravox model generation complete.")

    # Decode the generated text
    response_text = processor.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated text response: {response_text}")

    # Clean up memory
    del inputs
    del generated_ids
    if device != "cpu":
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
