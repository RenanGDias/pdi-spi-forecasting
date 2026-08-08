"""Transcreve a gravação da reunião usando faster-whisper localmente."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from faster_whisper import WhisperModel


def timestamp(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default="small")
    parser.add_argument("--model-dir", type=Path, default=Path("models/whisper"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/transcricao"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8",
        download_root=str(args.model_dir),
    )
    segments_iter, info = model.transcribe(
        str(args.audio),
        language="pt",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 350},
        word_timestamps=True,
        condition_on_previous_text=True,
    )

    segments = []
    for item in segments_iter:
        words = [
            {"start": word.start, "end": word.end, "text": word.word}
            for word in (item.words or [])
        ]
        segment = {
            "start": item.start,
            "end": item.end,
            "text": item.text.strip(),
            "words": words,
        }
        segments.append(segment)
        print(f"[{timestamp(item.start)}-{timestamp(item.end)}] {item.text.strip()}", flush=True)

    payload = {
        "audio": str(args.audio),
        "model": args.model,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": segments,
    }
    json_path = args.output_dir / "transcricao_bruta.json"
    md_path = args.output_dir / "transcricao_bruta.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_lines = [
        "# Transcrição bruta da reunião",
        "",
        f"- Modelo: `{args.model}`",
        f"- Idioma detectado: `{info.language}` ({info.language_probability:.1%})",
        f"- Duração analisada: `{timestamp(info.duration)}`",
        "",
        "> Transcrição automática ainda sem identificação confiável dos interlocutores.",
        "",
    ]
    md_lines.extend(
        f"**[{timestamp(segment['start'])}-{timestamp(segment['end'])}]** {segment['text']}"
        for segment in segments
    )
    md_path.write_text("\n\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
