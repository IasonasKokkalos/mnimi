"""The README's 60-second GIF: a real chat.py session (real BGE embedder, real CPU extractor,
real recall and export) with scripted assistant replies, rendered as terminal frames.

Usage: python tools/make_demo_gif.py <workdir> docs/mnimi-demo.gif [--session-only]
       python tools/make_demo_gif.py <workdir> docs/mnimi-demo.gif --render-only <transcript.json>
The session needs the [embed,extract] extras (the GGUF in the HF cache); the render needs Pillow.
The session is run once and its transcript saved beside the GIF, so the frames are the real
output; only the assistant's replies are scripted (no LLM backend in the loop).
"""
from __future__ import annotations

import datetime
import io
import json
import pathlib
import sys
from contextlib import redirect_stdout

try:  # the session venv (bge + llama) has no Pillow; the render venv has no llama
    from PIL import Image, ImageDraw, ImageFont
    FONT = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 17)
except ImportError:  # pragma: no cover
    Image = ImageDraw = ImageFont = FONT = None

sys.path.insert(0, r"D:\Projects\Personal\mnimi\examples")
sys.path.insert(0, r"D:\Projects\Personal\mnimi\src")

W, H = 960, 560
LINE_H = 22
PAD = 18
BG, FG, DIM = (24, 26, 32), (222, 224, 230), (130, 135, 150)
ACCENT, GREEN, LOG = (120, 180, 255), (150, 220, 150), (230, 180, 120)

SCRIPT = [  # (user text, scripted assistant reply)
    ("I'm Jason, I live in Eindhoven and I work on a memory library for LLM agents called mnimi.",
     "Nice to meet you, Jason. Noted: Eindhoven, and mnimi, a memory library for LLM agents."),
    ("My dentist appointment is next Thursday at 9. "
     "And I switched from PyCharm to VS Code last month.",
     "Got it: dentist next Thursday at 9, and VS Code is your editor now."),
    ("My sister's birthday is on the 24th of October. Remind me to call her.",
     "Noted: your sister's birthday is on October 24th. I will remind you to call her."),
]
QUERY = "when is my dentist appointment"


def run_session(workdir: pathlib.Path) -> list[dict]:
    """Drive chat.py's own functions; return the transcript as a list of {kind, text}."""
    import chat  # examples/chat.py

    db = str(workdir / "agent.db")
    day = datetime.date(2026, 10, 15)
    ts = chat.session_ts(day)
    memory = chat.build_memory(db, "bge", "cpu")
    replies = {u: a for u, a in SCRIPT}

    def backend(system: str, user: str) -> str:
        return replies[user]

    lines: list[dict] = []
    lines.append({"kind": "cmd",
                  "text": "$ python examples/chat.py --embedder bge --extractor cpu"})
    lines.append({"kind": "out", "text": f"mnimi chat: db=./agent.db user=me session ts={ts}; "
                                         "/recall <q>, /facts, /export, /consolidate, /quit"})
    last = ""
    for user, _ in SCRIPT:
        lines.append({"kind": "you", "text": user})
        buf = io.StringIO()
        with redirect_stdout(buf):
            reply = chat.turn(memory, "me", backend, user, ts)
        for extra in buf.getvalue().splitlines():
            lines.append({"kind": "log", "text": extra})
        lines.append({"kind": "bot", "text": reply})
        last = user
        buf = io.StringIO()
        with redirect_stdout(buf):
            chat.handle_command("/facts", memory, "me", db, last, ts)
        lines.append({"kind": "you", "text": "/facts"})
        for line in buf.getvalue().splitlines():
            lines.append({"kind": "out", "text": line})
    lines.append({"kind": "you", "text": f"/recall {QUERY}"})
    buf = io.StringIO()
    with redirect_stdout(buf):
        chat.handle_command(f"/recall {QUERY}", memory, "me", db, last, ts)
    for line in buf.getvalue().splitlines()[:4]:
        lines.append({"kind": "out", "text": line})
    lines.append({"kind": "you", "text": "/export"})
    buf = io.StringIO()
    with redirect_stdout(buf):
        chat.handle_command("/export", memory, "me", db, last, ts)
    for line in buf.getvalue().splitlines():
        lines.append({"kind": "out", "text": line.replace(str(workdir), ".")})
    lines.append({"kind": "cmd", "text": "$ cat memory.md"})
    md = (workdir / "memory.md").read_text(encoding="utf-8").splitlines()
    for line in md[:40]:
        lines.append({"kind": "md", "text": line})
    return lines


def wrap(text: str, width: int = 104) -> list[str]:
    out, cur = [], ""
    for word in text.split(" "):
        if len(cur) + len(word) + 1 > width and cur:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    return out + [cur]


def render(visible: list[tuple[str, str]], cursor: bool) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    max_lines = (H - 2 * PAD) // LINE_H
    rows = visible[-max_lines:]
    y = PAD
    for kind, text in rows:
        color = {"cmd": ACCENT, "you": GREEN, "bot": FG, "out": DIM, "log": LOG, "md": FG}[kind]
        d.text((PAD, y), text, font=FONT, fill=color)
        y += LINE_H
    if cursor:
        d.rectangle([PAD, y, PAD + 9, y + 18], fill=FG)
    return img


def frames_for(lines: list[dict]):
    """Yield (image, duration_ms). Typed lines appear in word chunks; output appears at once."""
    visible: list[tuple[str, str]] = []
    for item in lines:
        kind, text = item["kind"], item["text"]
        prefix = {"you": "you> ", "bot": "", "cmd": "", "out": "", "log": "", "md": ""}[kind]
        if kind in ("you", "cmd"):
            words = text.split(" ")
            for i in range(1, len(words) + 1):
                partial = prefix + " ".join(words[:i])
                yield render(visible + [(kind, w) for w in wrap(partial)], True), 70
            visible += [(kind, w) for w in wrap(prefix + text)]
            yield render(visible, False), 500
        elif kind == "bot":
            visible += [("bot", w) for w in wrap("    " + text)]
            yield render(visible, False), 1300
        elif kind == "log":
            visible += [("log", w) for w in wrap("  " + text)]
            yield render(visible, False), 900
        else:
            visible += [(kind, w) for w in wrap(text)]
            yield render(visible, False), 160
    yield render(visible, False), 3500


def main():
    workdir = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2])
    workdir.mkdir(parents=True, exist_ok=True)
    if "--render-only" in sys.argv:
        transcript = pathlib.Path(sys.argv[sys.argv.index("--render-only") + 1])
        lines = json.loads(transcript.read_text(encoding="utf-8"))
    else:
        lines = run_session(workdir)
        out.with_suffix(".transcript.json").write_text(
            json.dumps(lines, indent=1, ensure_ascii=False), encoding="utf-8")
        if "--session-only" in sys.argv:
            print(f"transcript: {len(lines)} lines")
            return
    imgs, durs = [], []
    for img, dur in frames_for(lines):
        imgs.append(img.quantize(colors=64, method=Image.Quantize.MEDIANCUT))
        durs.append(dur)
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=durs, loop=0, optimize=True)
    print(f"{out}: {len(imgs)} frames, {sum(durs)/1000:.1f} s, {out.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
