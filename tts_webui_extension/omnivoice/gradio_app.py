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
"""Gradio UI construction for OmniVoice extension."""

from typing import List, Optional

import gradio as gr

from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name
from tts_webui.utils.list_dir_models import unload_model_button

from .api import generate

# ---------------------------------------------------------------------------
# Language list — all 600+ supported languages
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)


# ---------------------------------------------------------------------------
# Voice Design instruction templates
# ---------------------------------------------------------------------------
_CATEGORIES = {
    "Gender / 性别": ["Male / 男", "Female / 女"],
    "Age / 年龄": [
        "Child / 儿童",
        "Teenager / 少年",
        "Young Adult / 青年",
        "Middle-aged / 中年",
        "Elderly / 老年",
    ],
    "Pitch / 音调": [
        "Very Low Pitch / 极低音调",
        "Low Pitch / 低音调",
        "Moderate Pitch / 中音调",
        "High Pitch / 高音调",
        "Very High Pitch / 极高音调",
    ],
    "Style / 风格": ["Whisper / 耳语"],
    "English Accent / 英文口音": [
        "American Accent / 美式口音",
        "Australian Accent / 澳大利亚口音",
        "British Accent / 英国口音",
        "Chinese Accent / 中国口音",
        "Canadian Accent / 加拿大口音",
        "Indian Accent / 印度口音",
        "Korean Accent / 韩国口音",
        "Portuguese Accent / 葡萄牙口音",
        "Russian Accent / 俄罗斯口音",
        "Japanese Accent / 日本口音",
    ],
    "Chinese Dialect / 中文方言": [
        "Henan Dialect / 河南话",
        "Shaanxi Dialect / 陕西话",
        "Sichuan Dialect / 四川话",
        "Guizhou Dialect / 贵州话",
        "Yunnan Dialect / 云南话",
        "Guilin Dialect / 桂林话",
        "Jinan Dialect / 济南话",
        "Shijiazhuang Dialect / 石家庄话",
        "Gansu Dialect / 甘肃话",
        "Ningxia Dialect / 宁夏话",
        "Qingdao Dialect / 青岛话",
        "Northeast Dialect / 东北话",
    ],
}

_ATTR_INFO = {
    "English Accent / 英文口音": "Only effective for English speech.",
    "Chinese Dialect / 中文方言": "Only effective for Chinese speech.",
}


# ---------------------------------------------------------------------------
# Reusable UI components
# ---------------------------------------------------------------------------


def _lang_dropdown(label: str = "Language (optional) / 语种 (可选)", value: str = "Auto"):
    return gr.Dropdown(
        label=label,
        choices=_ALL_LANGUAGES,
        value=value,
        allow_custom_value=False,
        interactive=True,
        info="Keep as Auto to auto-detect the language.",
    )


def _gen_settings():
    with gr.Accordion("Generation Settings (optional)", open=False):
        sp = gr.Slider(
            0.5,
            1.5,
            value=1.0,
            step=0.05,
            label="Speed",
            info="1.0 = normal. >1 faster, <1 slower. Ignored if Duration is set.",
        )
        du = gr.Number(
            value=None,
            label="Duration (seconds)",
            info=(
                "Leave empty to use speed."
                " Set a fixed duration to override speed."
            ),
        )
        ns = gr.Slider(
            4,
            64,
            value=32,
            step=1,
            label="Inference Steps",
            info="Default: 32. Lower = faster, higher = better quality.",
        )
        dn = gr.Checkbox(
            label="Denoise",
            value=True,
            info="Default: enabled. Uncheck to disable denoising.",
        )
        gs = gr.Slider(
            0.0,
            4.0,
            value=2.0,
            step=0.1,
            label="Guidance Scale (CFG)",
            info="Default: 2.0.",
        )
        pp = gr.Checkbox(
            label="Preprocess Prompt",
            value=True,
            info="apply silence removal and trimming to the reference "
            "audio, add punctuation in the end of reference text (if not already)",
        )
        po = gr.Checkbox(
            label="Postprocess Output",
            value=True,
            info="Remove long silences from generated audio.",
        )
    return ns, gs, dn, sp, du, pp, po


# ---------------------------------------------------------------------------
# UI builder
# ---------------------------------------------------------------------------


def ui():
    """Build the OmniVoice Gradio UI."""
    gr.Markdown(
        """
# OmniVoice

State-of-the-art text-to-speech model for **600+ languages**, supporting:

- **Voice Clone** — Clone any voice from a reference audio
- **Voice Design** — Create custom voices with speaker attributes

Built with [OmniVoice](https://github.com/k2-fsa/OmniVoice)
by Xiaomi Next-gen Kaldi team.
"""
    )

    with gr.Tabs():
        # ==============================================================
        # Voice Clone
        # ==============================================================
        with gr.TabItem("Voice Clone"):
            with gr.Row():
                with gr.Column(scale=1):
                    vc_text = gr.Textbox(
                        label="Text to Synthesize / 待合成文本",
                        lines=4,
                        placeholder="Enter the text you want to synthesize...",
                    )
                    vc_ref_audio = gr.Audio(
                        label="Reference Audio / 参考音频",
                        type="filepath",
                        elem_classes="compact-audio",
                    )
                    gr.Markdown(
                        "<span style='font-size:0.85em;color:#888;'>"
                        "Recommended: 3–10 seconds audio. "
                        "</span>"
                    )
                    vc_ref_text = gr.Textbox(
                        label=("Reference Text (optional)" " / 参考音频文本（可选）"),
                        lines=2,
                        placeholder="Transcript of the reference audio. Leave empty"
                        " to auto-transcribe via ASR models.",
                    )
                    vc_lang = _lang_dropdown("Language (optional) / 语种 (可选)")
                    (
                        vc_ns,
                        vc_gs,
                        vc_dn,
                        vc_sp,
                        vc_du,
                        vc_pp,
                        vc_po,
                    ) = _gen_settings()
                    vc_btn = gr.Button("Generate / 生成", variant="primary")
                with gr.Column(scale=1):
                    vc_audio = gr.Audio(
                        label="Output Audio / 合成结果",
                        type="numpy",
                    )
                    vc_status = gr.Textbox(label="Status / 状态", lines=2)

            def _clone_fn(
                text, lang, ref_aud, ref_text, ns, gs, dn, sp, du, pp, po
            ):
                result = generate(
                    text,
                    lang,
                    mode="clone",
                    ref_audio=ref_aud,
                    ref_text=ref_text,
                    num_step=ns,
                    guidance_scale=gs,
                    denoise=dn,
                    speed=sp,
                    duration=du,
                    preprocess_prompt=pp,
                    postprocess_output=po,
                )
                # result is dict with "audio_out" key
                audio_out = result.get("audio_out")
                status = "Done." if audio_out else "No audio generated."
                return audio_out, status

            vc_btn.click(
                _clone_fn,
                inputs=[
                    vc_text,
                    vc_lang,
                    vc_ref_audio,
                    vc_ref_text,
                    vc_ns,
                    vc_gs,
                    vc_dn,
                    vc_sp,
                    vc_du,
                    vc_pp,
                    vc_po,
                ],
                outputs=[vc_audio, vc_status],
            )

        # ==============================================================
        # Voice Design
        # ==============================================================
        with gr.TabItem("Voice Design"):
            with gr.Row():
                with gr.Column(scale=1):
                    vd_text = gr.Textbox(
                        label="Text to Synthesize / 待合成文本",
                        lines=4,
                        placeholder="Enter the text you want to synthesize...",
                    )
                    vd_lang = _lang_dropdown()

                    _AUTO = "Auto"
                    vd_groups = []
                    for _cat, _choices in _CATEGORIES.items():
                        vd_groups.append(
                            gr.Dropdown(
                                label=_cat,
                                choices=[_AUTO] + _choices,
                                value=_AUTO,
                                info=_ATTR_INFO.get(_cat),
                            )
                        )

                    (
                        vd_ns,
                        vd_gs,
                        vd_dn,
                        vd_sp,
                        vd_du,
                        vd_pp,
                        vd_po,
                    ) = _gen_settings()
                    vd_btn = gr.Button("Generate / 生成", variant="primary")
                with gr.Column(scale=1):
                    vd_audio = gr.Audio(
                        label="Output Audio / 合成结果",
                        type="numpy",
                    )
                    vd_status = gr.Textbox(label="Status / 状态", lines=2)

            def _build_instruct(groups: List[str]) -> Optional[str]:
                """Extract instruct text from UI dropdowns."""
                selected = [g for g in groups if g and g != "Auto"]
                if not selected:
                    return None
                parts = []
                for v in selected:
                    if " / " in v:
                        en, zh = v.split(" / ", 1)
                        # Dialects have no English equivalent
                        if "Dialect" in v.split(" / ")[0]:
                            parts.append(zh.strip())
                        else:
                            parts.append(en.strip())
                    else:
                        parts.append(v)
                return ", ".join(parts)

            def _design_fn(text, lang, ns, gs, dn, sp, du, pp, po, *groups):
                result = generate(
                    text,
                    lang,
                    mode="design",
                    instruct=_build_instruct(list(groups)),
                    num_step=ns,
                    guidance_scale=gs,
                    denoise=dn,
                    speed=sp,
                    duration=du,
                    preprocess_prompt=pp,
                    postprocess_output=po,
                )
                # result is dict with "audio_out" key
                audio_out = result.get("audio_out")
                status = "Done." if audio_out else "No audio generated."
                return audio_out, status

            vd_btn.click(
                _design_fn,
                inputs=[
                    vd_text,
                    vd_lang,
                    vd_ns,
                    vd_gs,
                    vd_dn,
                    vd_sp,
                    vd_du,
                    vd_pp,
                    vd_po,
                ]
                + vd_groups,
                outputs=[vd_audio, vd_status],
            )

        unload_model_button("omnivoice")

