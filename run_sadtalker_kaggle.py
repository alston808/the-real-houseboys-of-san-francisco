import os, sys, shutil, subprocess, pathlib

print("="*60)
print("  THE REAL HOUSEBOYS OF SF - SADTALKER GPU BATCH RUNNER")
print("="*60)

import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Active GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Warning: Running on CPU.")

WORK = pathlib.Path("/kaggle/working" if os.path.exists("/kaggle") else os.getcwd())
REPO_DIR = WORK / "the-real-houseboys-of-san-francisco"
SADTALKER_DIR = WORK / "SadTalker"
OUT_DIR = WORK / "outputs" / "sadtalker"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Clone repository assets if not present
if not REPO_DIR.exists() and not (WORK / "images").exists():
    print("Cloning repository assets...")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/alston808/the-real-houseboys-of-san-francisco.git", str(REPO_DIR)], check=True)
    IMAGES_DIR = REPO_DIR / "images"
    AUDIO_DIR = REPO_DIR / "audio"
elif (WORK / "images").exists():
    IMAGES_DIR = WORK / "images"
    AUDIO_DIR = WORK / "audio"
else:
    subprocess.run(["git", "-C", str(REPO_DIR), "pull"])
    IMAGES_DIR = REPO_DIR / "images"
    AUDIO_DIR = REPO_DIR / "audio"

# 2. Clone SadTalker if not present
if not SADTALKER_DIR.exists():
    print("Cloning SadTalker repository...")
    subprocess.run(["git", "clone", "https://github.com/OpenTalker/SadTalker.git", str(SADTALKER_DIR)], check=True)

# 3. Dependencies
print("Checking & installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy<2", "gradio", "imageio[ffmpeg]", "yacs", "pydub", "scipy", "scikit-image", "resampy", "librosa", "facexlib", "kornia", "basicsr", "gfpgan", "safetensors"], check=True)

# 4. Patch basicsr degradations.py
try:
    import basicsr
    bpath = pathlib.Path(basicsr.__file__).parent / "data" / "degradations.py"
    if bpath.exists():
        txt = bpath.read_text()
        if "functional_tensor" in txt:
            bpath.write_text(txt.replace("functional_tensor", "functional"))
            print("Patched basicsr degradations.py")
except Exception:
    pass

# 5. Patch SadTalker preprocess.py
prep_file = SADTALKER_DIR / "src" / "face3d" / "util" / "preprocess.py"
if prep_file.exists():
    content = prep_file.read_text()
    if "trans_params = np.array([w0, h0, s, t[0], t[1]])" in content:
        content = content.replace(
            "trans_params = np.array([w0, h0, s, t[0], t[1]])",
            "trans_params = np.array([float(w0), float(h0), float(np.squeeze(s)), float(np.squeeze(t[0])), float(np.squeeze(t[1]))])"
        )
        prep_file.write_text(content)
        print("Patched SadTalker preprocess.py")

# 6. Checkpoints setup
CKPT_DIR = SADTALKER_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
if not (CKPT_DIR / "BFM_Fitting").exists():
    print("Downloading 3DMM & model weights from Hugging Face...")
    subprocess.run(["git", "clone", "https://huggingface.co/vinthony/SadTalker", str(CKPT_DIR / "hf_models")], check=True)
    for item in (CKPT_DIR / "hf_models").glob("*"):
        target = CKPT_DIR / item.name
        if not target.exists():
            shutil.move(str(item), str(target))

if not (CKPT_DIR / "SadTalker_V0.0.2_256.safetensors").exists():
    print("Downloading SadTalker 256 safetensors...")
    subprocess.run(["wget", "-nc", "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors", "-O", str(CKPT_DIR / "SadTalker_V0.0.2_256.safetensors")], check=True)

GFPGAN_DIR = SADTALKER_DIR / "gfpgan" / "weights"
GFPGAN_DIR.mkdir(parents=True, exist_ok=True)
if not (GFPGAN_DIR / "GFPGANv1.4.pth").exists():
    subprocess.run(["wget", "-nc", "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth", "-O", str(GFPGAN_DIR / "GFPGANv1.4.pth")], check=True)

# 7. Render all 18 confessionals
CHARACTERS = ["prest", "felix", "marco", "dex", "river", "ash"]
completed = []

print("\n=== STARTING ANIMATION RENDERING ===")
for char in CHARACTERS:
    portrait = IMAGES_DIR / f"{char}.png"
    if not portrait.exists():
        print(f"Skipping {char}: portrait not found at {portrait}")
        continue
    print(f"\n🎬 Processing Character: {char.upper()}")
    for beat in [1, 2, 3]:
        audio = AUDIO_DIR / f"{char}_confessional_{beat}.wav"
        if not audio.exists():
            candidates = list(AUDIO_DIR.glob(f"{char}*{beat}*.wav"))
            if candidates:
                audio = candidates[0]
            else:
                print(f"  Warning: Audio not found for {char} beat {beat}")
                continue
        out_mp4 = OUT_DIR / f"{char}_confessional_{beat}.mp4"
        if out_mp4.exists():
            print(f"  ✓ Already generated: {out_mp4.name}")
            completed.append(str(out_mp4))
            continue
        print(f"  -> Rendering beat {beat}: {audio.name}...")
        cmd = [
            sys.executable, str(SADTALKER_DIR / "inference.py"),
            "--driven_audio", str(audio),
            "--source_image", str(portrait),
            "--result_dir", str(OUT_DIR),
            "--still",
            "--preprocess", "crop",
            "--size", "256",
            "--checkpoint_dir", str(CKPT_DIR)
        ]
        if not torch.cuda.is_available():
            cmd.append("--cpu")
        subprocess.run(cmd, cwd=str(SADTALKER_DIR), check=True)

        recent = sorted(list(OUT_DIR.glob("*.mp4")), key=lambda p: os.path.getmtime(p))
        if recent and recent[-1] != out_mp4:
            shutil.move(str(recent[-1]), str(out_mp4))
        print(f"  ✓ Created: {out_mp4.name}")
        completed.append(str(out_mp4))

# 8. Archive to ZIP
zip_archive = WORK / "sadtalker_confessionals_all"
shutil.make_archive(str(zip_archive), "zip", str(OUT_DIR))
final_zip = WORK / "sadtalker_confessionals_all.zip"
print("\n" + "="*60)
print(f"✓ COMPLETE! Rendered {len(completed)} clips.")
print(f"✓ Zip file saved to: {final_zip}")
print("="*60)
