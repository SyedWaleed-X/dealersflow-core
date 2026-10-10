import os
import subprocess
import tempfile
import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app
from backend.core.database import engine


@pytest.fixture(autouse=True)
async def cleanup_db_pool():
    yield
    await engine.dispose()


def generate_sample_wav() -> str:
    """Generates a clean 3-second 16kHz test WAV using ffmpeg."""
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav_path = temp_wav.name
    temp_wav.close()

    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "sine=frequency=1000:duration=3",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        temp_wav_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    return temp_wav_path


@pytest.mark.anyio
async def test_end_to_end_voice_lead_pipeline():
    """
    Verifies complete end-to-end flow:
    Audio Upload -> Faster-Whisper -> Qwen/Gemini -> CatBoost -> pgvector -> WhatsApp Link.
    """
    sample_wav = generate_sample_wav()

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with open(sample_wav, "rb") as f:
                files = {"file": ("test_voice.wav", f, "audio/wav")}
                response = await client.post("/api/leads/process-voice", files=files)

            # Assertions
            assert response.status_code == 201, f"E2E Pipeline failed with {response.status_code}: {response.text}"
            data = response.json()

            assert "property_id" in data
            assert "lead" in data
            assert "valuation" in data
            assert "whatsapp_deep_link" in data
            assert data["valuation"]["estimated_price_pkr"] > 0
            assert "https://wa.me/" in data["whatsapp_deep_link"]

            print(f"\n[+] E2E Pipeline Verified! Created Property ID: {data['property_id']}")
            print(f"[+] Valuation Generated: {data['valuation']['estimated_price_pkr']:,} PKR")

    finally:
        if os.path.exists(sample_wav):
            os.remove(sample_wav)