import torch
import gc
import librosa
import tempfile
import torchaudio
import gradio as gr
from transformers import AutoProcessor, AutoModel
from chatterbox.tts import ChatterboxTTS

# 1. Ultravox model and processor loading
try:
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
        device_map="auto"
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

    # The turns are where the actual conversation history is built.
    # The Ultravox model's purpose is to act as the AI.
    # We pass the audio *with* the roles, and the model generates the response for the last turn.
    # The model expects the 'user' turn to have content relevant to the audio.
    # Let's try a simpler turns structure first, and let the model handle the speech.

    # According to Ultravox's example, 'turns' provides the conversational context.
    # The model itself will process the audio as the "user" input for the *current* turn.
    # The 'Please respond to my speech input' is an instruction, not a placeholder for transcription.
    # The generated text *is* the model's response to the spoken input + the system turn.
    turns = [
        {"role": "system", "content": "You are a helpful voice assistant."},
        # The content for the user turn when audio is provided is typically empty or a placeholder
        # and the model uses the audio to "fill in" this turn's meaning for its response generation.
        # Let's keep it as is, as the model should handle audio within this context.
        # If the model expects specific formatting for audio-linked turns, that's defined in its remote code.
        {"role": "user", "content": ""} # User content is the audio itself.
    ]
    print("Conversation turns prepared.")

    # Process audio and generate text using the loaded model
    # Crucially, let's try to explicitly add attention_mask to fix the warning.
    inputs = processor(
        audio=audio,
        sampling_rate=sr,
        turns=turns,
        return_tensors="pt"
    )

    # Adding attention mask manually if processor doesn't do it perfectly (it should for 'pt' tensors)
    # The warning suggests it's related to pad_token==eos_token
    # This might require checking the processor's output keys.
    # A common way to get attention_mask is from the processor's output if it's there.
    if "attention_mask" not in inputs and "input_ids" in inputs:
        # Create a simple attention mask if not provided by processor: 1 for tokens, 0 for padding
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print("Inputs processed and moved to model device.")

    # Generate the output from the model
    with torch.no_grad():
        print("Generating response from Ultravox model...")
        # The model's custom generate method in its remote code handles the audio.
        # We need to make sure the model is actually performing *speech-to-text* then *text-to-text*.
        # The `generated_ids` should be the *textual response* from the LLM part of Ultravox.
        generated_ids = model.generate(**inputs, max_new_tokens=128)

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
