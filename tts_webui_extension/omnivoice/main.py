import gradio as gr

from .gradio_app import ui as build_ui


def omnivoice_ui():
    build_ui()


def extension__tts_generation_webui():
    omnivoice_ui()

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
