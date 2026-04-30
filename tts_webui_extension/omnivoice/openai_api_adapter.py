import os

from tts_webui_extension.omnivoice.api import generate


def register():
    try:
        if os.environ.get("OPENAI_PROXY_HOST"):
            register_unsafe_outprocess()
        else:
            register_unsafe_inprocess()
    except Exception as e:
        print(f"Error registering OmniVoice API adapter: {e}")
        print("OmniVoice TTS will not be available on the OpenAI API.")


def register_unsafe_inprocess():
    from tts_webui_extension.openai_tts_api.services.tts_service import register_tts_adapter
    from tts_webui_extension.openai_tts_api.services.voice_service import register_voice_getter

    def tts_fn(
        model: str, text: str, voice: str | None, speed: float | None, params: dict
    ) -> dict:
        return generate(
            text=text,
            mode="clone" if voice else "design",
            ref_audio=voice,
            speed=speed,
            **params,
        )

    register_tts_adapter("omnivoice", tts_fn)
    register_voice_getter("omnivoice", lambda: [])


def register_unsafe_outprocess():
    from tts_webui_extension.openai_tts_api.harness import setup_oai_server

    def tts_fn(
        model: str, text: str, voice: str | None, speed: float | None, params: dict
    ) -> dict:
        return generate(
            text=text,
            mode="clone" if voice else "design",
            ref_audio=voice,
            speed=speed,
            **params,
        )

    # register_with and port are resolved from OPENAI_PROXY_HOST / GRADIO_SERVER_PORT env vars
    setup_oai_server(
        tts_fn=tts_fn,
        get_voices_fn=lambda model: [],
        model="omnivoice",
    )
