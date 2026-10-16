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


def wrap(text: str, width: int = 98) -> list[str]:  # 98 × 9 px fits the 924 px text area
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


def group(lines: list[dict]):
    """Consecutive output lines of one kind form a block; the hold after it scales with its size."""
    block: list[dict] = []
    for item in lines:
        if item["kind"] in ("out", "md") and (not block or block[-1]["kind"] == item["kind"]):
            block.append(item)
            continue
        if block:
            yield block
            block = []
        if item["kind"] in ("out", "md"):
            block.append(item)
        else:
            yield [item]
    if block:
        yield block


def hold_for(rows: int, per_row: int = 900, lo: int = 2000, hi: int = 6000) -> int:
    return max(lo, min(hi, per_row * rows))


def frames_for(lines: list[dict]):
    """Yield (image, duration_ms). Pacing: typed lines appear word by word; a reply or an output
    block appears line by line, then holds long enough to be read (about 0.9 s per rendered row).
    The Markdown dump starts on a cleared screen and holds once more when it fills the screen."""
    visible: list[tuple[str, str]] = []
    max_lines = (H - 2 * PAD) // LINE_H
    for block in group(lines):
        kind = block[0]["kind"]
        if kind in ("you", "cmd"):
            text = block[0]["text"]
            prefix = "you> " if kind == "you" else ""
            if text.startswith("$ cat "):
                visible = []  # the export is read on a fresh screen, as after `clear`
            words = text.split(" ")
            for i in range(1, len(words) + 1):
                partial = prefix + " ".join(words[:i])
                yield render(visible + [(kind, w) for w in wrap(partial)], True), 70
            visible += [(kind, w) for w in wrap(prefix + text)]
            yield render(visible, False), 600
        elif kind == "bot":
            rows = wrap("    " + block[0]["text"])
            visible += [("bot", w) for w in rows]
            yield render(visible, False), hold_for(len(rows), lo=2000)
        elif kind == "log":
            rows = wrap("  " + block[0]["text"])
            visible += [("log", w) for w in rows]
            yield render(visible, False), 1200
        else:  # an output block: "out" after a command, "md" for the dump
            rows_total = 0
            filled_hold_done = False
            for item in block:
                rows = wrap(item["text"])
                visible += [(kind, w) for w in rows]
                rows_total += len(rows)
                step = 150 if kind == "md" else 180
                if kind == "md" and not filled_hold_done and len(visible) >= max_lines:
                    # the first screenful of the export is readable before it scrolls
                    yield render(visible, False), hold_for(max_lines, per_row=250, lo=3500, hi=6000)
                    filled_hold_done = True
                else:
                    yield render(visible, False), step
            if kind == "md":
                yield render(visible, False), 6500
            else:
                yield render(visible, False), hold_for(rows_total)


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
