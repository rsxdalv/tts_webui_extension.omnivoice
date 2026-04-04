import gradio as gr

from .api import load_model
from .gradio_app import ui as build_ui


def ui_wrapper():
    """Mount the OmniVoice UI into the parent app."""
    build_ui()


def omnivoice_ui():
    gr.Markdown(
        """
    # Omnivoice
    
    State-of-the-art text-to-speech model for **600+ languages**, supporting:
    
    - **Voice Clone** — Clone any voice from a reference audio
    - **Voice Design** — Create custom voices with speaker attributes
    
    Built with [OmniVoice](https://github.com/k2-fsa/OmniVoice)
    by Xiaomi Next-gen Kaldi team.
    """
    )


def extension__tts_generation_webui():
    omnivoice_ui()

    # Load the model when extension is initialized
    try:
        load_model("k2-fsa/OmniVoice")
    except Exception as e:
        print(f"Warning: Could not preload OmniVoice model: {e}")

    return {
        "package_name": "tts_webui_extension.omnivoice",
        "name": "Omnivoice",
        "requirements": "git+https://github.com/rsxdalv/tts_webui_extension.omnivoice@main",
        "description": "State-of-the-art massive multilingual zero-shot text-to-speech model supporting 600+ languages with voice cloning and voice design.",
        "extension_type": "interface",
        "extension_class": "text-to-speech",
        "author": "Xiaomi Next-gen Kaldi team",
        "extension_author": "rsxdalv",
        "license": "MIT",
        "website": "https://github.com/k2-fsa/OmniVoice",
        "extension_website": "https://github.com/rsxdalv/tts_webui_extension.omnivoice",
        "extension_platform_version": "0.0.1",
    }


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()
    with gr.Blocks() as demo:
        with gr.Tab("Omnivoice", id="omnivoice"):
            omnivoice_ui()

    demo.launch(
        server_port=7772,
    )
