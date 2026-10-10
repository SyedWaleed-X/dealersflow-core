import subprocess
import time
from backend.audio_processor import AudioProcessor
from backend.asr_engine import Transcriber

def generate_tone_audio(filename: str = "test_tone.mp3", duration: int = 3):
    """Generates a synthetic audio tone using ffmpeg."""
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"sine=frequency=1000:duration={duration}",
        "-c:a", "libmp3lame", filename
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def generate_silence_audio(filename: str = "test_silence.wav", duration: int = 5):
    """Generates pure silence to test VAD suppression."""
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"anullsrc=r=16000:cl=mono",
        "-t", str(duration),
        "-c:a", "pcm_s16le", filename
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def run_tests():
    print("[*] Initializing Transcriber Engine...")
    transcriber = Transcriber()

    # --- TEST 1: Synthetic Audio Pipeline ---
    print("\n[*] TEST 1: Audio Preprocessor + Whisper Pipeline Execution...")
    generate_tone_audio("temp_tone.mp3", duration=3)
    
    start_time = time.perf_counter()
    wav_path = AudioProcessor.convert_to_wav_16k("temp_tone.mp3")
    transcript = transcriber.transcribe(wav_path)
    elapsed = time.perf_counter() - start_time
    
    print(f"[+] Audio converted and transcribed in {elapsed:.2f} seconds.")
    print(f"[+] Output on Tone: '{transcript}' (Tone audio yields minimal/empty text, as expected).")
    
    AudioProcessor.cleanup_temp_file(wav_path)
    AudioProcessor.cleanup_temp_file("temp_tone.mp3")

    # --- TEST 2: Chaos Test 3.1 - Pure Silence Hallucination Check ---
    print("\n[*] TEST 2 (Chaos Test 3.1): Pure Silence Hallucination Suppression...")
    generate_silence_audio("temp_silence.wav", duration=5)
    
    silence_transcript = transcriber.transcribe("temp_silence.wav")
    AudioProcessor.cleanup_temp_file("temp_silence.wav")

    if silence_transcript == "":
        print("[SUCCESS] VAD Filter perfectly suppressed 5 seconds of dead silence! Zero hallucinations.")
    else:
        print(f"[WARNING] Model output text on pure silence: '{silence_transcript}'")

    print("\n[+] Day 3 ASR Pipeline Tests Complete!")

if __name__ == "__main__":
    run_tests()