import asyncio
import collections
import functools
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx
import mlx_whisper  # type: ignore
import numpy as np
import sounddevice as sd  # type: ignore
from dotenv import load_dotenv

from blacki.adk_runtime import SessionLocator, create_adk_runtime
from blacki.container import close_container, init_container
from blacki.utils.config import ServerEnv, initialize_environment

# Audio Recording Settings
SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000
SILENCE_THRESHOLD = 0.02


async def play_beep(frequency: float = 1000.0, duration: float = 0.15) -> None:
    """Plays a simple beep tone to acknowledge wake word."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    # Generate a sine wave
    tone = np.sin(frequency * t * 2 * np.pi)
    # Apply a quick envelope to prevent clicking clicks at start/end
    envelope = np.ones_like(tone)
    fade_len = int(SAMPLE_RATE * 0.02)
    if fade_len > 0:
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    tone = tone * envelope * 0.3  # Scale volume down

    await asyncio.get_running_loop().run_in_executor(
        None, functools.partial(sd.play, tone, samplerate=SAMPLE_RATE, blocking=True)
    )


async def listen_for_wake_word(stt_model_path: str) -> str:
    """Continuously listen for the wake word ('blacki') using a rolling buffer."""
    print("🎧 Listening for wake word ('Blacki')...")

    buffer_duration = 2.5
    check_interval = 0.5
    max_frames = int(SAMPLE_RATE * buffer_duration)

    audio_buffer: collections.deque[np.ndarray] = collections.deque()
    buffer_frames = 0

    loop = asyncio.get_running_loop()

    def callback(
        indata: np.ndarray, frames: int, time: Any, status: sd.CallbackFlags
    ) -> None:
        if status:
            print(status, file=sys.stderr)

        nonlocal buffer_frames
        chunk = indata.copy()
        audio_buffer.append(chunk)
        buffer_frames += frames

        while buffer_frames > max_frames and len(audio_buffer) > 1:
            first_chunk = audio_buffer[0]
            if buffer_frames - len(first_chunk) >= max_frames:
                buffer_frames -= len(first_chunk)
                audio_buffer.popleft()
            else:
                break

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback)
    with stream:
        while True:
            await asyncio.sleep(check_interval)

            if buffer_frames < int(
                SAMPLE_RATE * 1.0
            ):  # Wait until we have at least 1 second
                continue

            # Safely copy the current buffer
            current_audio = (
                np.concatenate(list(audio_buffer), axis=0)
                if audio_buffer
                else np.array([])
            )

            if len(current_audio) == 0:
                continue

            # 1. Energy Gate: Skip pure silence
            rms = np.sqrt(np.mean(current_audio**2))
            if rms < SILENCE_THRESHOLD:
                continue

            # Transcribe chunk
            result = await loop.run_in_executor(
                None,
                functools.partial(
                    mlx_whisper.transcribe,
                    current_audio.flatten(),
                    path_or_hf_repo=stt_model_path,
                ),
            )

            # 2. no_speech_prob check
            segments = result.get("segments", [])
            if segments:
                avg_no_speech = sum(
                    s.get("no_speech_prob", 0.0) for s in segments
                ) / len(segments)
                if avg_no_speech > 0.6:
                    continue

            text = result["text"].lower().strip()
            # Remove punctuation to ensure easy matching
            text = re.sub(r"[^\w\s]", "", text)

            # 3. Hallucination Guard
            words = text.split()
            if words and len(set(words)) == 1 and len(words) > 3:
                continue

            if "blacki" in text or "blacky" in text:
                print(
                    f"🌟 Wake word detected! (Heard: '{str(result['text']).strip()}')"
                )
                await play_beep()
                return str(result["text"]).strip()


async def record_command_until_silence(silence_duration: float = 1.5) -> np.ndarray:
    """Records audio from microphone until silence is detected."""
    print("🎧 Listening for command... (Speak anytime)")

    q: asyncio.Queue[np.ndarray] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def callback(
        indata: np.ndarray, frames: int, time: Any, status: sd.CallbackFlags
    ) -> None:
        if status:
            print(status, file=sys.stderr)
        # using call_soon_threadsafe to put in queue
        loop.call_soon_threadsafe(q.put_nowait, indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback)

    recording = []
    pre_speech_buffer: collections.deque[np.ndarray] = collections.deque()
    pre_speech_buffer_frames = 0
    max_pre_speech_frames = int(SAMPLE_RATE * 0.5)

    has_spoken = False
    silent_frames = 0
    total_frames = 0
    max_wait_frames = int(SAMPLE_RATE * 10.0)

    with stream:
        while True:
            indata = await q.get()
            frames = len(indata)
            total_frames += frames

            # calculate energy (RMS)
            rms = np.sqrt(np.mean(indata**2))

            if rms > SILENCE_THRESHOLD:
                has_spoken = True
                silent_frames = 0
                recording.append(indata)
            elif has_spoken:
                silent_frames += frames
                recording.append(indata)
            else:
                pre_speech_buffer.append(indata)
                pre_speech_buffer_frames += frames
                while (
                    pre_speech_buffer_frames > max_pre_speech_frames
                    and len(pre_speech_buffer) > 1
                ):
                    popped = pre_speech_buffer.popleft()
                    pre_speech_buffer_frames -= len(popped)

                if total_frames > max_wait_frames:
                    print("⏳ No speech detected, timing out.")
                    break

            if has_spoken and silent_frames > int(SAMPLE_RATE * silence_duration):
                break

    print("⏹️ Command ended. Processing...")

    if not has_spoken:
        return np.array([])

    final_audio = list(pre_speech_buffer) + recording
    if not final_audio:
        return np.array([])
    return np.concatenate(final_audio, axis=0)


async def stream_audio_response(tts_client: httpx.AsyncClient, text: str) -> None:
    """Streams audio directly from Custom TTS to the speakers."""
    stream = sd.RawOutputStream(samplerate=TTS_SAMPLE_RATE, channels=1, dtype="int16")
    stream.start()
    try:
        files = {
            "text": (None, text),
            "voice": (None, "alba"),
            "output_format": (None, "wav"),
        }

        header_skipped = False
        buffer = b""

        async with tts_client.stream(
            "POST", "/api/synthesize/stream", files=files
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=4096):
                if not header_skipped:
                    buffer += chunk
                    if len(buffer) >= 44:
                        # Skip the 44-byte WAV header
                        pcm_chunk = buffer[44:]
                        header_skipped = True
                        if pcm_chunk:
                            await asyncio.get_running_loop().run_in_executor(
                                None, stream.write, pcm_chunk
                            )
                else:
                    await asyncio.get_running_loop().run_in_executor(
                        None, stream.write, chunk
                    )
    except httpx.HTTPError as e:
        print(f"\n❌ HTTP Error streaming TTS: {e}")
    except Exception as e:
        print(f"\n❌ Error streaming TTS: {e}")
    finally:
        stream.stop()
        stream.close()


async def main() -> None:
    load_dotenv()

    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not chat_id:
        print(
            "⚠️ Warning: TELEGRAM_CHAT_ID not found in .env. "
            "Using default standalone session."
        )

    # Initialize Blacki ADK Runtime
    env = initialize_environment(ServerEnv)

    # Initialize global container so tools that depend on SQLite can function
    container = None
    sqlite_path = env.sqlite_path or str(Path(env.agent_dir) / ".adk" / "tools.db")
    container = await init_container(sqlite_path)
    await container.initialize_all_storages()

    runtime = create_adk_runtime(env)

    if chat_id:
        # Emulate Telegram Session Identity exactly as bot.py does
        conversation_key = f"chat-{chat_id}"
        stable_identity = f"telegram-{conversation_key}"
        locator = SessionLocator(
            user_id=stable_identity,
            session_id_prefix=stable_identity,
        )
        state = {
            "user_id": stable_identity,
            "telegram_chat_id": chat_id,
            "telegram_conversation_key": conversation_key,
        }
    else:
        # Generate a default standalone session if no chat ID is provided
        locator = SessionLocator(
            user_id="speech-client",
            session_id_prefix="speech-client",
        )
        state = {"user_id": "speech-client"}

    # Initialize STT (mlx-whisper for native Apple MPS acceleration)
    print("Loading MLX Whisper model (small) and warming up MPS...")
    stt_model_path = "mlx-community/whisper-small-mlx"

    # Warm up model to avoid initial delay
    await asyncio.get_running_loop().run_in_executor(
        None,
        functools.partial(
            mlx_whisper.transcribe,
            np.zeros(SAMPLE_RATE, dtype=np.float32),
            path_or_hf_repo=stt_model_path,
        ),
    )

    # Initialize TTS (Custom via httpx)
    tts_base_url = os.getenv("TTS_BASE_URL", "http://localhost:8000")
    tts_client = httpx.AsyncClient(base_url=tts_base_url, timeout=httpx.Timeout(60.0))

    print("\n✅ Speech Client Ready!")

    try:
        print("🔊 Testing Agent and TTS with initial greeting...")
        print("🧠 Agent is thinking...")
        agent_response = await runtime.run_user_turn(
            locator=locator, message_text="Hi", state=state
        )
        print(f"\n🤖 Blacki (Initial): {agent_response}")
        await stream_audio_response(tts_client, agent_response)

        while True:
            # Always record command directly
            audio_data = await record_command_until_silence()

            if len(audio_data) < 1000:
                print("Audio too short, skipping...")
                continue

            # Transcribe
            print("⏳ Transcribing...")
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                functools.partial(
                    mlx_whisper.transcribe,
                    audio_data.flatten(),
                    path_or_hf_repo=stt_model_path,
                ),
            )
            user_text = result["text"].strip()

            if not user_text:
                print("Could not hear anything clearly.")
                continue

            print(f"\n🗣️ You: {user_text}")

            # Agent Turn
            print("🧠 Agent is thinking...")
            agent_response = await runtime.run_user_turn(
                locator=locator, message_text=user_text, state=state
            )
            print(f"\n🤖 Blacki: {agent_response}")

            # Text-to-Speech Streaming
            print("🔊 Streaming audio response...")
            await stream_audio_response(tts_client, agent_response)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        await tts_client.aclose()
        await runtime.close()
        if container is not None:
            await close_container()


if __name__ == "__main__":
    asyncio.run(main())
