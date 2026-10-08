#!/usr/bin/env python3
"""
Batch SadTalker runner for local CPU execution on all 18 confessionals.
Skips clips that have already been generated.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SADTALKER_DIR = ROOT / "SadTalker"
PYTHON_BIN = SADTALKER_DIR / ".venv" / "bin" / "python"
IMAGES_DIR = ROOT / "images"
AUDIO_DIR = ROOT / "audio"
OUT_DIR = ROOT / "outputs" / "sadtalker"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHARACTERS = ["prest", "felix", "marco", "dex", "river", "ash"]

print("=" * 60)
print("  THE REAL HOUSEBOYS OF SF - LOCAL SADTALKER BATCH RUNNER")
print("=" * 60)

completed = []

for char in CHARACTERS:
    portrait = IMAGES_DIR / f"{char}.png"
    if not portrait.exists():
        print(f"Skipping {char}: {portrait} not found")
        continue

    print(f"\n🎬 Character: {char.upper()}")
    for beat in [1, 2, 3]:
        audio = AUDIO_DIR / f"{char}_confessional_{beat}.wav"
        if not audio.exists():
            candidates = list(AUDIO_DIR.glob(f"{char}*{beat}*.wav"))
            if candidates:
                audio = candidates[0]
            else:
                print(f"  Warning: No audio for {char} beat {beat}")
                continue

        target_mp4 = OUT_DIR / f"{char}_confessional_{beat}.mp4"
        if target_mp4.exists():
            print(f"  ✓ Already generated: {target_mp4.name}")
            completed.append(target_mp4)
            continue

        print(f"  -> Rendering beat {beat}: {audio.name}...")
        cmd = [
            str(PYTHON_BIN), "inference.py",
            "--driven_audio", str(audio),
            "--source_image", str(portrait),
            "--result_dir", str(OUT_DIR),
            "--still",
            "--preprocess", "crop",
            "--size", "256",
            "--checkpoint_dir", "checkpoints",
            "--cpu"
        ]

        res = subprocess.run(cmd, cwd=str(SADTALKER_DIR))
        if res.returncode == 0:
            # Locate newly produced mp4 and rename to target
            mp4_candidates = sorted(
                list(OUT_DIR.glob("*.mp4")),
                key=lambda p: os.path.getmtime(p)
            )
            # Find the most recently modified mp4 that isn't already a named target
            for candidate in reversed(mp4_candidates):
                if candidate != target_mp4 and not any(candidate == c for c in completed):
                    shutil.move(str(candidate), str(target_mp4))
                    break

            if target_mp4.exists():
                print(f"  ✓ Successfully created: {target_mp4.name}")
                completed.append(target_mp4)
        else:
            print(f"  ✗ Failed rendering {char} beat {beat} (code {res.returncode})")

print("\n" + "=" * 60)
print(f"Finished! Total clips completed: {len(completed)} / 18")
print(f"All outputs located in: {OUT_DIR}")
print("=" * 60)
