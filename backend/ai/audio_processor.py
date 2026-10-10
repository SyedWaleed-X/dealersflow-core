import os
import subprocess
import tempfile
from pathlib import Path


class AudioProcessingError(Exception):
    """Custom exception raised when audio conversion fails."""
    pass


class AudioProcessor:
    """
    Standardizes arbitrary incoming audio files (WhatsApp .ogg, .mp3, .m4a)
    into a clean 16,000 Hz, 16-bit, single-channel (mono) PCM WAV file.
    """

    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1

    @classmethod
    def convert_to_wav_16k(cls, input_audio_path: str | Path) -> str:
        """
        Converts any input audio file to standard 16kHz mono PCM WAV using ffmpeg.
        Returns the path to the newly created temporary WAV file.
        """
        input_path = Path(input_audio_path)

        # 1. Input Validation
        if not input_path.exists():
            raise AudioProcessingError(f"Audio file not found: {input_path}")
        
        if input_path.stat().st_size == 0:
            raise AudioProcessingError(f"Audio file is empty (0 bytes): {input_path}")

        # 2. Prepare Unique Temporary Output File
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_wav_path = temp_wav.name
        temp_wav.close()

        # 3. Build FFmpeg Command:
        # -y : Overwrite output
        # -i : Input file path
        # -ar 16000 : Resample to 16,000 Hz
        # -ac 1 : Downmix to 1 channel (mono)
        # -c:a pcm_s16le : 16-bit uncompressed linear PCM
        # -loglevel error : Suppress noisy banner logs
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-ar", str(cls.TARGET_SAMPLE_RATE),
            "-ac", str(cls.TARGET_CHANNELS),
            "-c:a", "pcm_s16le",
            "-loglevel", "error",
            str(output_wav_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )

            if result.returncode != 0:
                # Cleanup failed temp file
                if os.path.exists(output_wav_path):
                    os.remove(output_wav_path)
                raise AudioProcessingError(
                    f"FFmpeg conversion failed: {result.stderr.strip()}"
                )

            return output_wav_path

        except FileNotFoundError:
            raise AudioProcessingError(
                "FFmpeg executable not found in system PATH. Ensure FFmpeg is installed."
            )

    @staticmethod
    def cleanup_temp_file(file_path: str | Path) -> None:
        """Safely removes temporary audio artifacts after processing."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass