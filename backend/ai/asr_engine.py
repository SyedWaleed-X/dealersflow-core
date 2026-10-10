import os
import sys
import logging
from pathlib import Path

# --- AUTO-DETECT PIP CUDA 12 DLLS ON WINDOWS ---
if sys.platform == "win32":
    try:
        import nvidia.cublas.lib
        import nvidia.cudnn.lib

        cublas_dir = os.path.dirname(nvidia.cublas.lib.__file__)
        cudnn_dir = os.path.dirname(nvidia.cudnn.lib.__file__)

        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(cublas_dir)
            os.add_dll_directory(cudnn_dir)

        os.environ["PATH"] = cublas_dir + os.pathsep + cudnn_dir + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass

from faster_whisper import WhisperModel
from backend.core.config import settings

logger = logging.getLogger(__name__)

ROMAN_URDU_INITIAL_PROMPT = (
    "Assalam o Alaikum, main DHA phase 6 mein 1 kanal plot bechna chahta hoon. "
    "Demand 4 crore 50 lac hai. Direct owner tariq 03001234567. "
    "Buyer ya seller intent, cash payment, LDA standard marla kanal."
)


class Transcriber:
    """
    Offline Speech-to-Text engine powered by Faster-Whisper.
    Biases decoding toward Latin-script Roman-Urdu via autoregressive prompt conditioning.
    Runs on RTX 3060 CUDA GPU with automatic graceful fallback to CPU.
    """

    def __init__(self):
        self.model_size = settings.whisper_model_size
        self.device = settings.whisper_device
        self.compute_type = settings.whisper_compute_type

        self._load_model()

    def _load_model(self, force_cpu: bool = False):
        if force_cpu:
            self.device = "cpu"
            self.compute_type = "int8"
            self.model_size = "small" if self.model_size == "large-v3-turbo" else self.model_size
            print(f"[!] Falling back: Loading Whisper [{self.model_size}] on [CPU int8]...")
        else:
            print(f"[*] Loading Whisper [{self.model_size}] on [{self.device}] with [{self.compute_type}]...")

        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type
        )

    def transcribe(
        self,
        audio_wav_path: str | Path,
        force_roman_urdu: bool = True
    ) -> str:
        audio_path_str = str(audio_wav_path)
        prompt = ROMAN_URDU_INITIAL_PROMPT if force_roman_urdu else None

        try:
            return self._execute_transcribe(audio_path_str, prompt)
        except RuntimeError as e:
            if "cublas" in str(e).lower() or "cuda" in str(e).lower():
                print(f"\n[!] WARNING: GPU execution issue ({e}).")
                print("[!] Self-healing: Switching to CPU int8 mode dynamically...")
                self._load_model(force_cpu=True)
                return self._execute_transcribe(audio_path_str, prompt)
            raise e

    def _execute_transcribe(self, audio_path_str: str, prompt: str | None) -> str:
        segments, info = self.model.transcribe(
            audio_path_str,
            beam_size=5,
            initial_prompt=prompt,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            task="transcribe"
        )

        transcript_chunks = [segment.text.strip() for segment in segments]
        return " ".join(transcript_chunks).strip()