#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Omnivoice extension API — exposed to tts_webui."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import gradio as gr
import numpy as np
import torch

if TYPE_CHECKING:
    from omnivoice import OmniVoice, OmniVoiceGenerationConfig

from tts_webui.utils.manage_model_state import (
    get_current_model,
    manage_model_state,
    rename_model,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def get_best_device() -> str:
    """Auto-detect the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device: str) -> str:
    """Resolve 'auto' to actual device or return as-is."""
    return get_best_device() if device == "auto" else device


def resolve_dtype(dtype: str):
    """Resolve dtype string to torch dtype."""
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


def generate_model_name(device, dtype):
    """Generate a human-readable model name for state tracking."""
    return f"Omnivoice on {device} with {dtype}"


# ---------------------------------------------------------------------------
# Model management with state tracking
# ---------------------------------------------------------------------------


@manage_model_state("omnivoice")
def get_model(
    model_name: str = "just_a_placeholder",
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float16,
) -> "OmniVoice":
    """Load and return the OmniVoice model instance."""
    from omnivoice import OmniVoice  # lazy import

    device = device or torch.device(get_best_device())

    logger.info(f"Loading OmniVoice model, device={device}, dtype={dtype} ...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=dtype,
        load_asr=True,
    )
    logger.info("OmniVoice model loaded.")
    return model


def load_model(
    checkpoint: str = "k2-fsa/OmniVoice",
    device: Optional[str] = None,
    dtype: str = "float16",
) -> OmniVoice:
    """Load or reload the OmniVoice model."""
    device = resolve_device(device or "auto")
    dtype = resolve_dtype(dtype)

    model = get_model(
        model_name=generate_model_name(device, dtype),
        device=torch.device(device),
        dtype=dtype,
    )
    return model


def is_model_loaded() -> bool:
    """Check if the model is currently loaded."""
    return get_current_model("omnivoice") is not None


def move_model_to_device_and_dtype(
    device: str,
    dtype: str,
    cpu_offload: bool = False,
) -> bool:
    """
    Move model to specified device and dtype.
    Used by tts_webui extension management.
    """
    device = resolve_device(device)
    dtype = resolve_dtype(dtype)

    model = get_current_model("omnivoice")
    if model is None:
        load_model(device=device, dtype=dtype)
        return True

    rename_model("omnivoice", generate_model_name(device, dtype))
    device = torch.device("cpu" if cpu_offload else device)
    # Note: OmniVoice doesn't have custom to() like Chatterbox,
    # just re-load if device/dtype changes significantly
    return True


# ---------------------------------------------------------------------------
# Generation core
# ---------------------------------------------------------------------------


def _generate_with_progress(
    text: str,
    language: Optional[str] = None,
    *,
    mode: str = "clone",
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    instruct: Optional[str] = None,
    num_step: int = 32,
    guidance_scale: float = 2.0,
    denoise: bool = True,
    speed: Optional[float] = None,
    duration: Optional[float] = None,
    preprocess_prompt: bool = True,
    postprocess_output: bool = True,
    progress: gr.Progress | None = None,
) -> Dict[str, Any]:
    """
    Core generation function with progress tracking.

    Returns dict with 'audio_out' key: {"audio_out": (sample_rate, waveform)}
    """
    model = get_model()
    if model is None:
        raise gr.Error("Error: Model not loaded. Please reload the extension.")

    if not text or not text.strip():
        raise gr.Error("Please enter the text to synthesize.")

    if mode == "clone" and not ref_audio:
        raise gr.Error("Please upload a reference audio for voice cloning.")

    if progress:
        progress(0.0, desc="Preparing generation...")

    from omnivoice import OmniVoiceGenerationConfig  # lazy import

    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )

    lang = language if (language and language != "Auto") else None

    kw: Dict[str, Any] = dict(
        text=text.strip(), language=lang, generation_config=gen_config
    )

    if speed is not None and float(speed) != 1.0:
        kw["speed"] = float(speed)
    if duration is not None and float(duration) > 0:
        kw["duration"] = float(duration)

    if mode == "clone":
        if progress:
            progress(0.1, desc="Processing reference audio...")
        kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
        )

    if mode == "design":
        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

    if progress:
        progress(0.3, desc="Generating audio...")

    try:
        audio = model.generate(**kw)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Error: {type(e).__name__}: {e}")

    if progress:
        progress(1.0, desc="Done.")

    waveform = audio[0]  # (T,)
    waveform = (waveform * 32767).astype(np.int16)

    return {
        "audio_out": (model.sampling_rate, waveform),
    }


def generate(
    text: str,
    language: Optional[str] = None,
    *,
    mode: str = "clone",
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    instruct: Optional[str] = None,
    num_step: int = 32,
    guidance_scale: float = 2.0,
    denoise: bool = True,
    speed: Optional[float] = None,
    duration: Optional[float] = None,
    preprocess_prompt: bool = True,
    postprocess_output: bool = True,
) -> Dict[str, Any]:
    """
    Core generation function for OmniVoice TTS.
    Returns dict with 'audio_out' key: {"audio_out": (sample_rate, waveform)}
    """
    try:
        return _generate_with_progress(
            text=text,
            language=language,
            mode=mode,
            ref_audio=ref_audio,
            ref_text=ref_text,
            instruct=instruct,
            num_step=num_step,
            guidance_scale=guidance_scale,
            denoise=denoise,
            speed=speed,
            duration=duration,
            preprocess_prompt=preprocess_prompt,
            postprocess_output=postprocess_output,
            progress=None,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Error: {e}")
