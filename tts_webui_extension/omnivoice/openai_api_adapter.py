import numpy as np

from tts_webui_extension.omnivoice.api import generate


def register():
    try:
        register_unsafe()
    except Exception as e:
        print(f"Error registering OmniVoice API adapter: {e}")
        print("OmniVoice TTS will not be available on the OpenAI API.")

def register_unsafe():
    from tts_webui_extension.openai_tts_api.services.tts_service import (
        register_tts_adapter,
        register_tts_streaming_adapter,
    )

    def tts_adapter(
        model: str, text: str, voice: str | None, speed: float | None, params: dict
    ) -> dict:
        return generate(
            text=text,
            mode="clone" if voice else "design",
            ref_audio=voice,
            speed=speed,
            **params,
        )

    register_tts_adapter("omnivoice", tts_adapter)

    from tts_webui_extension.openai_tts_api.services.voice_service import (
        register_voice_getter,
    )

    def voice_getter() -> list[dict]:
        return []

    register_voice_getter("omnivoice", voice_getter)
