import torch
import gc
import librosa
import tempfile
import torchaudio
import gradio as gr
import numpy as np
import asyncio
import threading
from transformers import pipeline
from chatterbox.tts import ChatterboxTTS
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")

# Global variables for models - load once, use many times
uv_pipe = None
tts_model = None
executor = ThreadPoolExecutor(max_workers=2)

def initialize_models():
    """Initialize models once at startup"""
    global uv_pipe, tts_model
    
    print(f"Using device: {'cuda:0' if torch.cuda.is_available() else 'cpu'}")
    
    # 1. Load Ultravox pipeline with optimizations
    try:
        print("Loading Ultravox pipeline...")
        uv_pipe = pipeline(
            model="fixie-ai/ultravox-v0_4",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,  # Always use float16 for speed
            model_kwargs={
                "attn_implementation": "flash_attention_2",  # Use flash attention if available
                "use_cache": True,
                "low_cpu_mem_usage": True,
            }
        )
        print("Ultravox pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading Ultravox pipeline: {e}")
        # Fallback without flash attention
        uv_pipe = pipeline(
            model="fixie-ai/ultravox-v0_4",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
            model_kwargs={
                "use_cache": True,
                "low_cpu_mem_usage": True,
            }
        )
        print("Ultravox pipeline loaded successfully (fallback mode).")
    
    # 2. Load Chatterbox TTS
    try:
        print("Loading Chatterbox TTS...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        print("Chatterbox TTS loaded successfully.")
    except Exception as e:
        print(f"Error loading Chatterbox TTS: {e}")
        exit()

class OptimizedTTS:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def synthesize(self, text: str) -> str:
        """Optimized TTS synthesis"""
        if not text.strip():
            # Return silence for empty text
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            silence = torch.zeros(1, 1600)  # 0.1 seconds of silence
            torchaudio.save(tmp.name, silence, 16000)
            return tmp.name

        # Generate audio with optimizations
        with torch.inference_mode():  # Disable gradient computation
            wav = tts_model.generate(text)
            
        if not isinstance(wav, torch.Tensor):
            wav = torch.from_numpy(wav)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
            
        # Save to temporary file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        torchaudio.save(tmp.name, wav.cpu(), tts_model.sr)
        
        return tmp.name

# Initialize optimized TTS
tts = OptimizedTTS()

def preprocess_audio(audio_path: str) -> tuple:
    """Optimized audio preprocessing"""
    try:
        # Load audio with librosa (faster than torchaudio for this use case)
        audio, sr = librosa.load(audio_path, sr=16000, mono=True, dtype=np.float32)
        
        # Basic audio processing optimizations
        if len(audio) == 0:
            return None, None
            
        # Trim silence from beginning and end
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        return audio, sr
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return None, None

def generate_response(audio: np.ndarray, sr: int) -> str:
    """Optimized response generation"""
    try:
        # Prepare conversation context (minimal for speed)
        turns = [{
            "role": "system", 
            "content": "You are a helpful voice assistant. Be concise and natural."
        }]
        
        # Generate response with optimized parameters
        with torch.inference_mode():
            result = uv_pipe({
                'audio': audio,
                'turns': turns,
                'sampling_rate': sr
            }, 
            max_new_tokens=64,  # Reduced for faster generation
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=uv_pipe.tokenizer.eos_token_id
            )
        
        # Extract response text
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and "generated_text" in result[0]:
                response_text = result[0]["generated_text"]
            else:
                response_text = str(result[0])
        elif isinstance(result, dict) and "generated_text" in result:
            response_text = result["generated_text"]
        else:
            response_text = str(result)
            
        return response_text.strip()
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm sorry, I couldn't process that. Could you try again?"

def s2s_optimized(audio_path: str) -> str:
    """Optimized speech-to-speech function"""
    if audio_path is None:
        print("No audio input received.")
        return None

    print(f"Processing audio from: {audio_path}")
    
    # Step 1: Preprocess audio
    audio, sr = preprocess_audio(audio_path)
    if audio is None:
        print("Failed to load audio")
        return None
        
    print(f"Audio loaded with sample rate: {sr}, duration: {len(audio)/sr:.2f}s")

    # Step 2: Generate response (ASR + LLM)
    print("Generating response...")
    response_text = generate_response(audio, sr)
    print(f"Generated response: {response_text}")
    
    # Step 3: Synthesize speech
    print("Synthesizing speech...")
    synthesized_audio_path = tts.synthesize(response_text)
    print(f"Speech synthesis complete: {synthesized_audio_path}")
    
    # Minimal memory cleanup (only when necessary)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return synthesized_audio_path

# Async version for even better performance (optional)
async def s2s_async(audio_path: str) -> str:
    """Async version of s2s for better concurrency"""
    loop = asyncio.get_event_loop()
    
    # Run in thread pool to avoid blocking
    result = await loop.run_in_executor(executor, s2s_optimized, audio_path)
    return result

# Gradio interface with optimizations
def create_interface():
    """Create optimized Gradio interface"""
    
    def process_audio_sync(audio_path):
        """Synchronous wrapper for Gradio"""
        return s2s_optimized(audio_path)
    
    with gr.Blocks(
        title="UltraChat S2S Agent - Optimized",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 800px !important;
            margin: auto !important;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🚀 UltraChat: High-Speed Speech-to-Speech AI
        **Optimized for low latency on A40 GPU**
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    sources=["microphone"], 
                    type="filepath", 
                    label="🎤 Your Speech",
                    format="wav"
                )
                
                process_btn = gr.Button(
                    "🗣️ Talk to AI", 
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=1):
                audio_out = gr.Audio(
                    type="filepath", 
                    label="🤖 AI Response",
                    autoplay=True
                )
        
        # Status indicator
        status = gr.Textbox(
            label="Status", 
            value="Ready", 
            interactive=False,
            max_lines=1
        )
        
        # Event handlers
        def update_status_processing():
            return "Processing..."
            
        def update_status_ready():
            return "Ready"
        
        # Chain the events
        process_btn.click(
            fn=update_status_processing,
            outputs=status
        ).then(
            fn=process_audio_sync,
            inputs=audio_in,
            outputs=audio_out
        ).then(
            fn=update_status_ready,
            outputs=status
        )
        
        # Also trigger on audio upload
        audio_in.change(
            fn=update_status_processing,
            outputs=status
        ).then(
            fn=process_audio_sync,
            inputs=audio_in,
            outputs=audio_out
        ).then(
            fn=update_status_ready,
            outputs=status
        )
    
    return demo

def main():
    """Main function with proper initialization"""
    print("🚀 Initializing UltraChat S2S Agent (Optimized)...")
    
    # Initialize models once at startup
    initialize_models()
    
    # Warm up the models with dummy data
    print("Warming up models...")
    try:
        # Create dummy audio for warmup
        dummy_audio = np.random.randn(16000).astype(np.float32) * 0.01  # 1 second of quiet noise
        dummy_response = generate_response(dummy_audio, 16000)
        dummy_tts = tts.synthesize("Hello")
        print("Model warmup complete!")
    except Exception as e:
        print(f"Warmup failed (non-critical): {e}")
    
    # Create and launch interface
    demo = create_interface()
    
    print("🎉 Launching optimized Gradio demo...")
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False,
        inbrowser=False,
        quiet=True,
        show_error=True,
        server_kwargs={
            "threaded": True,
        }
    )

if __name__ == "__main__":
    main()
