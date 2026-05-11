import asyncio
import functools
import os
import sys
import termios
import tty
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


def wait_for_spacebar(prompt: str) -> None:
    """Wait for the user to press the spacebar."""
    print(prompt, end="", flush=True)
    if not sys.stdin.isatty():
        # Handle cases where stdin is not a terminal (e.g., piped or backgrounded)
        sys.stdin.readline()
        return

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        # Flush any pending input so it doesn't instantly trigger
        termios.tcflush(fd, termios.TCIFLUSH)
        while True:
            char = sys.stdin.read(1)
            if char == " ":
                print()
                break
            elif char == "\x03":  # Ctrl+C
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


async def record_audio_until_spacebar() -> np.ndarray:
    """Records audio from microphone until the user hits Spacebar."""
    await asyncio.get_event_loop().run_in_executor(
        None, wait_for_spacebar, "\n🎤 Press [SPACEBAR] to start recording..."
    )

    recording = []

    def callback(
        indata: np.ndarray, frames: int, time: Any, status: sd.CallbackFlags
    ) -> None:
        if status:
            print(status, file=sys.stderr)
        recording.append(indata.copy())

    print("🔴 Recording... (Press [SPACEBAR] to stop)")
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback)
    with stream:
        await asyncio.get_event_loop().run_in_executor(None, wait_for_spacebar, "")

    print("⏹️ Stopped recording. Processing...")
    if not recording:
        return np.array([])
    return np.concatenate(recording, axis=0)


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
                            await asyncio.get_event_loop().run_in_executor(
                                None, stream.write, pcm_chunk
                            )
                else:
                    await asyncio.get_event_loop().run_in_executor(
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

    # Initialize global container so tools that depend on Postgres can function
    container = None
    if env.database_url:
        container = await init_container(env.database_url)
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
    print("Loading MLX Whisper model (small)...")
    stt_model_path = "mlx-community/whisper-small-mlx"

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
            # 1. Record Audio
            audio_data = await record_audio_until_spacebar()

            if len(audio_data) < 1000:
                print("Audio too short, skipping...")
                continue

            # 2. Transcribe
            print("⏳ Transcribing...")
            result = await asyncio.get_event_loop().run_in_executor(
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

            # 3. Agent Turn
            print("🧠 Agent is thinking...")
            agent_response = await runtime.run_user_turn(
                locator=locator, message_text=user_text, state=state
            )
            print(f"\n🤖 Blacki: {agent_response}")

            # 4. Text-to-Speech Streaming
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
