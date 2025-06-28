import torch
import gc
import librosa
import tempfile
import torchaudio
import gradio as gr
from transformers import AutoProcessor, AutoModel
from chatterbox.tts import ChatterboxTTS # Ensure this import is correct and visible

# 1. Ultravox model and processor loading
try:
    # Determine the target device. device_map will handle final placement.
    # We still keep 'device' variable for inputs processing.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the processor
    processor = AutoProcessor.from_pretrained(
        "fixie-ai/ultravox-v0_4",
        trust_remote_code=True,
        revision="main"
    )
    print("Ultravox processor loaded successfully.")

    # Load the model with device_map="auto" to handle memory placement
    model = AutoModel.from_pretrained(
        "fixie-ai/ultravox-v0_4",
        trust_remote_code=True,
        revision="main",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" # IMPORTANT: This tells transformers/accelerate to handle device placement
    )
    print("Ultravox model loaded successfully with device_map='auto'.")

except Exception as e:
    print(f"Error loading Ultravox model: {e}")
    print("This error usually indicates a memory issue (not enough VRAM/RAM)")
    print("or an incompatibility with how the model's custom code interacts with Transformers/PyTorch's loading mechanism.")
    print("1. Check your GPU memory usage (`nvidia-smi`).")
    print("2. Ensure `accelerate` is installed and up-to-date (`pip install --upgrade accelerate`).")
    print("3. Consider if your environment has enough RAM for initial loading.")
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
            # The NameError suggests ChatterboxTTS wasn't properly seen.
            # It's imported at the top, so this *should* work. If not, it implies
            # an unusual module loading order or environment issue.
            # Assuming the top-level import makes ChatterboxTTS available:
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
        # Ensure wav is on CPU before saving with torchaudio
        torchaudio.save(tmp.name, wav.cpu(), self.model.sr)
        # Clear GPU cache
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        return tmp.name

tts = TTS() # Keep this after the TTS class definition

# 3. Speech-to-speech function
def s2s(audio_path: str) -> str:
    # --- FIX FOR "Invalid file: None" ---
    if audio_path is None:
        print("No audio input received yet. Waiting for user interaction.")
        # Return an empty string or a placeholder audio path for initial Gradio setup
        # or raise a more specific error if an actual audio file is always expected.
        # For Gradio's initial run, returning None for audio_out is fine.
        return None # Return None for audio output if no input. Gradio handles this gracefully.
    # --- END FIX ---

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
    )
    # Move inputs to the device where the model is loaded (which device_map="auto" determined)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print("Inputs processed and moved to model device.")

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
    if torch.cuda.is_available(): # Check if CUDA is available before trying to empty cache
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
    # For initial load, set audio_out to None.
    # The fn=s2s will be called on button click, not immediately for the initial state.
    btn.click(fn=s2s, inputs=audio_in, outputs=audio_out)

if __name__ == "__main__":
    print("Launching Gradio demo...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
