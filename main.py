import torch
import gc
import librosa
import tempfile
import torchaudio
import gradio as gr
from transformers import pipeline
from chatterbox.tts import ChatterboxTTS
import threading
import time

# Set optimizations for A40 GPU
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 1. Optimized Ultravox pipeline
try:
    print(f"Using device: {'cuda:0' if torch.cuda.is_available() else 'cpu'}")
    
    uv_pipe = pipeline(
        model="fixie-ai/ultravox-v0_4",
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,  # Always use float16 for speed
        model_kwargs={
            "attn_implementation": "flash_attention_2",  # Use flash attention if available
            "use_cache": True,
        }
    )
    print("Ultravox pipeline loaded successfully.")

except Exception as e:
    print(f"Error loading Ultravox pipeline: {e}")
    # Fallback without flash attention
    try:
        uv_pipe = pipeline(
            model="fixie-ai/ultravox-v0_4",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        print("Ultravox pipeline loaded successfully (fallback mode).")
    except Exception as e2:
        print(f"Fallback also failed: {e2}")
        exit()

# 2. Highly optimized TTS class
class FastTTS:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.warmup_done = False

    def load(self):
        if self.model is None:
            print(f"Loading Chatterbox TTS on device: {self.device}")
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            print("Chatterbox TTS loaded successfully.")
            self._warmup()
        return True

    def _warmup(self):
        """Warmup the model with a short sentence to avoid first-call latency"""
        if not self.warmup_done:
            print("Warming up TTS model...")
            try:
                _ = self.model.generate("Hi")
                self.warmup_done = True
                print("TTS warmup complete.")
            except:
                pass

    def synthesize(self, text: str, max_length: int = 200) -> str:
        self.load()
        
        if not text.strip():
            print("Warning: Received empty text for TTS, returning empty audio.")
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            torchaudio.save(tmp.name, torch.zeros(1, 16000), 16000)
            return tmp.name

        # Truncate very long text to reduce latency
        if len(text) > max_length:
            # Find the last complete sentence within the limit
            truncated = text[:max_length]
            last_period = truncated.rfind('.')
            if last_period > max_length // 2:  # If we find a period reasonably close to the end
                text = truncated[:last_period + 1]
            else:
                text = truncated + "..."
            print(f"Truncated text to: {text}")

        # Generate with optimized settings
        start_time = time.time()
        
        # Use faster sampling if available
        try:
            wav = self.model.generate(
                text,
                # Add any speed optimization parameters here
                # These depend on the ChatterboxTTS implementation
            )
        except:
            wav = self.model.generate(text)
        
        generation_time = time.time() - start_time
        print(f"TTS generation took: {generation_time:.2f}s")

        if not isinstance(wav, torch.Tensor):
            wav = torch.from_numpy(wav)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        torchaudio.save(tmp.name, wav.cpu(), self.model.sr)
        
        # Quick cleanup
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return tmp.name

# Initialize TTS
tts = FastTTS()

# Pre-load TTS in background thread
def preload_tts():
    tts.load()

threading.Thread(target=preload_tts, daemon=True).start()

# 3. Optimized speech-to-speech function
def s2s(audio_path: str) -> str:
    if audio_path is None:
        print("No audio input received yet. Waiting for user interaction.")
        return None

    total_start = time.time()
    print(f"Processing audio from: {audio_path}")
    
    # Load audio
    audio_start = time.time()
    audio, sr = librosa.load(audio_path, sr=16000)
    print(f"Audio loading took: {time.time() - audio_start:.2f}s")

    # Prepare conversation with shorter system prompt for faster processing
    turns = [
        {
            "role": "system",
            "content": "You are a helpful voice assistant. Be concise and natural."
        }
    ]

    # Run Ultravox inference
    inference_start = time.time()
    result = uv_pipe({
        'audio': audio, 
        'turns': turns, 
        'sampling_rate': sr
    }, 
    max_new_tokens=64,  # Reduced for faster response
    do_sample=True,
    temperature=0.7,
    pad_token_id=uv_pipe.tokenizer.eos_token_id
    )
    inference_time = time.time() - inference_start
    print(f"Ultravox inference took: {inference_time:.2f}s")
    
    # Extract response text
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
    
    print(f"Generated text: {response_text}")

    # Quick GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # TTS synthesis with length limit for speed
    tts_start = time.time()
    synthesized_audio_path = tts.synthesize(response_text, max_length=150)
    tts_time = time.time() - tts_start
    
    total_time = time.time() - total_start
    print(f"Total pipeline time: {total_time:.2f}s (Inference: {inference_time:.2f}s, TTS: {tts_time:.2f}s)")
    
    return synthesized_audio_path

# 4. Optimized Gradio interface
with gr.Blocks(
    title="UltraChat S2S Agent",
    theme=gr.themes.Soft(),
    css="footer {visibility: hidden}"
) as demo:
    gr.Markdown("# 🎙️ UltraChat: Speech-to-Speech AI Agent")
    gr.Markdown("*Optimized for low latency on A40 GPU*")
    
    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(
                sources=["microphone"], 
                type="filepath", 
                label="Your Speech",
                streaming=False,  # Disable streaming for better performance
            )
            btn = gr.Button("▶️ Talk to AI", variant="primary")
        
        with gr.Column():
            audio_out = gr.Audio(
                type="filepath", 
                label="AI Response",
                autoplay=True  # Auto-play the response
            )
    
    # Status display
    status = gr.Textbox(label="Status", interactive=False, visible=False)
    
    btn.click(
        fn=s2s, 
        inputs=audio_in, 
        outputs=audio_out,
        show_progress=True
    )

if __name__ == "__main__":
    print("Launching optimized Gradio demo...")
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        show_error=True,
        quiet=False
    )
