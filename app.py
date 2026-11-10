#!/usr/bin/env python3
"""
PapaVoice Cloud TTS Server (2025-2026 Production Edition for Render)
Supports:
  1. Auto-download from GitHub Release URL (MODEL_ZIP_URL)
  2. Auto-extraction from local repo ZIP (papa_voice_MORNING_READY.zip)
  3. Direct PC-to-Render Browser ZIP Upload via Web UI
"""
import os
import shutil
import zipfile
import tempfile
import subprocess
import urllib.request
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

# =====================================================================
#                        PATHS & CONFIGURATION
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent
PIPER_BIN = BASE_DIR / "piper_engine" / "piper" / "piper"
MODEL_DIR = BASE_DIR / "active_model"
ONNX_PATH = MODEL_DIR / "papa_voice.onnx"
JSON_PATH = MODEL_DIR / "papa_voice.onnx.json"

MODEL_ZIP_URL = os.environ.get("MODEL_ZIP_URL", "").strip()
ACTIVE_SOURCE_NAME = "Not Loaded"


def extract_zip_to_model(zip_path: Path, source_label: str) -> bool:
    """Extracts papa_voice.onnx and papa_voice.onnx.json from any ZIP file without manual unzipping."""
    global ACTIVE_SOURCE_NAME
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    found_onnx = False
    found_json = False

    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.endswith(".onnx") and not name.endswith(".onnx.json"):
                with z.open(name) as src, open(ONNX_PATH, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                found_onnx = True
            elif name.endswith(".onnx.json"):
                with z.open(name) as src, open(JSON_PATH, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                found_json = True

    if found_onnx and found_json and ONNX_PATH.exists() and JSON_PATH.exists():
        ACTIVE_SOURCE_NAME = source_label
        mb = ONNX_PATH.stat().st_size / 1e6
        print(f"✅ PapaVoice Model Extracted & Ready! ({ACTIVE_SOURCE_NAME} -> {mb:.1f} MB)", flush=True)
        return True
    return False


def prepare_voice_model():
    """Loads voice model automatically on Render server startup."""
    global ACTIVE_SOURCE_NAME
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. If MODEL_ZIP_URL is set in Render Environment Variables, download & extract it
    if MODEL_ZIP_URL and (not ONNX_PATH.exists() or not JSON_PATH.exists()):
        try:
            print(f"🌐 Downloading model ZIP from MODEL_ZIP_URL: {MODEL_ZIP_URL}", flush=True)
            dl_zip = BASE_DIR / "downloaded_release_model.zip"
            req = urllib.request.Request(MODEL_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as response, open(dl_zip, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
            extract_zip_to_model(dl_zip, "GitHub Release ZIP")
            dl_zip.unlink(missing_ok=True)
        except Exception as e:
            print(f"⚠️ MODEL_ZIP_URL download warning: {e}", flush=True)

    # 2. Check if a .zip file exists in the repository root
    zips = sorted(BASE_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if zips and (not ONNX_PATH.exists() or not JSON_PATH.exists()):
        extract_zip_to_model(zips[0], zips[0].name)

    # 3. Check if loose .onnx and .onnx.json files exist in repository root
    loose_onnx = BASE_DIR / "papa_voice.onnx"
    loose_json = BASE_DIR / "papa_voice.onnx.json"
    if loose_onnx.exists() and loose_json.exists() and not ONNX_PATH.exists():
        shutil.copy2(loose_onnx, ONNX_PATH)
        shutil.copy2(loose_json, JSON_PATH)
        ACTIVE_SOURCE_NAME = "Direct Repo ONNX"

    if ONNX_PATH.exists() and JSON_PATH.exists():
        mb = ONNX_PATH.stat().st_size / 1e6
        print(f"🎉 Server Ready with Model: {ACTIVE_SOURCE_NAME} ({mb:.1f} MB)", flush=True)
    else:
        print("ℹ️ Model not loaded yet. Set MODEL_ZIP_URL or upload ZIP via the Web UI.", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    prepare_voice_model()
    yield


app = FastAPI(
    title="PapaVoice Hindi TTS Cloud API",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Hindi text to speak")
    length_scale: float = Field(1.05, ge=0.5, le=2.0, description="Speed (1.0 = normal, >1.0 = slower)")


def synthesize_wav_bytes(text: str, length_scale: float = 1.05) -> bytes:
    if not ONNX_PATH.exists() or not JSON_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model load nahi hua hai! Niche 'PC se ZIP Upload Karein' button se papa_voice_MORNING_READY.zip upload karein."
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)

    try:
        cmd = [
            str(PIPER_BIN),
            "--model", str(ONNX_PATH),
            "--config", str(JSON_PATH),
            "--output_file", str(tmp_wav),
            "--length_scale", str(length_scale),
            "--noise_scale", "0.667",
            "--noise_w", "0.8",
        ]
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{PIPER_BIN.parent}:{env.get('LD_LIBRARY_PATH', '')}"

        proc = subprocess.run(
            cmd,
            input=text.strip().encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(PIPER_BIN.parent),
            env=env,
            timeout=30,
            check=False,
        )

        if proc.returncode != 0 or not tmp_wav.exists() or tmp_wav.stat().st_size < 100:
            err = proc.stderr.decode("utf-8", errors="ignore")
            raise HTTPException(status_code=500, detail=f"Piper synthesis error: {err}")

        return tmp_wav.read_bytes()
    finally:
        tmp_wav.unlink(missing_ok=True)


# =====================================================================
#                        API & WEB ROUTES
# =====================================================================
@app.get("/health")
def health_check():
    ready = ONNX_PATH.exists() and JSON_PATH.exists()
    return {
        "status": "ok" if ready else "waiting_for_model",
        "model_ready": ready,
        "active_source": ACTIVE_SOURCE_NAME
    }


@app.get("/api/model-info")
def model_info():
    ready = ONNX_PATH.exists() and JSON_PATH.exists()
    size_mb = round(ONNX_PATH.stat().st_size / 1e6, 2) if ready else 0.0
    return {
        "model_ready": ready,
        "source_pack": ACTIVE_SOURCE_NAME,
        "onnx_size_mb": size_mb,
        "language": "hi_IN"
    }


@app.post("/api/upload-model")
async def upload_model_zip(file: UploadFile = File(...)):
    """Allows uploading papa_voice_MORNING_READY.zip directly from PC browser to Render."""
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Kripya sirf .zip file (papa_voice_MORNING_READY.zip) select karein!")

    tmp_zip = BASE_DIR / "uploaded_pc_model.zip"
    try:
        with open(tmp_zip, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        ok = extract_zip_to_model(tmp_zip, file.filename)
        if not ok:
            raise HTTPException(status_code=400, detail="ZIP ke andar .onnx aur .onnx.json file nahi mili!")
        return {"status": "success", "active_source": ACTIVE_SOURCE_NAME}
    finally:
        tmp_zip.unlink(missing_ok=True)


@app.post("/api/tts")
def tts_post(req: TTSRequest):
    wav_data = synthesize_wav_bytes(req.text, req.length_scale)
    return Response(content=wav_data, media_type="audio/wav")


@app.get("/api/tts")
def tts_get(
    text: str = Query(..., min_length=1, max_length=1000),
    length_scale: float = Query(1.05, ge=0.5, le=2.0)
):
    wav_data = synthesize_wav_bytes(text, length_scale)
    return Response(content=wav_data, media_type="audio/wav")


@app.get("/", response_class=HTMLResponse)
def web_ui():
    ready = ONNX_PATH.exists() and JSON_PATH.exists()
    badge = f"✅ Model Ready ({ACTIVE_SOURCE_NAME})" if ready else "⚠️ Model Not Loaded Yet (Upload ZIP Below)"
    badge_color = "#a6e3a1" if ready else "#f38ba8"

    return f"""
    <!DOCTYPE html>
    <html lang="hi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PapaVoice Cloud Studio</title>
        <style>
            body {{
                background: #1e1e2e; color: #cdd6f4; font-family: system-ui, sans-serif;
                display: flex; justify-content: center; padding: 25px 15px; margin: 0;
            }}
            .card {{
                background: #181825; max-width: 580px; width: 100%; border-radius: 16px;
                padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.4);
            }}
            h2 {{ margin-top: 0; color: #89b4fa; }}
            .badge {{
                display: inline-block; padding: 6px 14px; border-radius: 20px;
                font-size: 0.88rem; font-weight: bold; background: #313244; color: {badge_color};
                margin-bottom: 16px;
            }}
            textarea {{
                width: 100%; box-sizing: border-box; height: 115px; border-radius: 10px;
                border: 1px solid #45475a; background: #313244; color: #fff;
                padding: 12px; font-size: 1.1rem; resize: vertical;
            }}
            .controls {{ margin: 14px 0; display: flex; align-items: center; justify-content: space-between; }}
            button {{
                width: 100%; background: #89b4fa; color: #11111b; border: none;
                padding: 14px; font-size: 1.05rem; font-weight: bold; border-radius: 10px;
                cursor: pointer; transition: 0.2s;
            }}
            button:hover {{ background: #b4befe; }}
            button:disabled {{ opacity: 0.6; cursor: wait; }}
            audio {{ width: 100%; margin-top: 18px; }}
            #status {{ margin-top: 10px; font-size: 0.92rem; color: #f9e2af; }}
            .upload-box {{
                margin-top: 22px; padding-top: 16px; border-top: 1px solid #313244;
                font-size: 0.9rem; color: #a6adc8;
            }}
            .upload-btn {{
                background: #a6e3a1; margin-top: 8px; padding: 10px; font-size: 0.95rem;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <h2>🎙️ PapaVoice Hindi AI Studio</h2>
            <div id="badge" class="badge">{badge}</div>
            <textarea id="text">नमस्ते बेटा, मैं बिल्कुल ठीक हूँ और हमेशा तुम्हारे साथ हूँ।</textarea>
            <div class="controls">
                <label for="speed">Bolne ki Speed (1.05 = Natural):</label>
                <input type="number" id="speed" value="1.05" step="0.05" min="0.6" max="1.6"
                       style="width:70px; padding:6px; border-radius:6px; border:none;">
            </div>
            <button id="btn" onclick="generateVoice()">🔊 आवाज़ बनाएँ और सुनें</button>
            <div id="status"></div>
            <audio id="player" controls style="display:none;"></audio>

            <div class="upload-box">
                <strong>📂 Direct PC से Model ZIP बदलें / अपलोड करें:</strong><br>
                <input type="file" id="zipFile" accept=".zip" style="margin-top:8px; width:100%;">
                <button class="upload-btn" id="upBtn" onclick="uploadZip()">⬆️ PC से papa_voice_MORNING_READY.zip अपलोड करें</button>
            </div>
        </div>
        <script>
            async function generateVoice() {{
                const text = document.getElementById('text').value.trim();
                const speed = parseFloat(document.getElementById('speed').value) || 1.05;
                const btn = document.getElementById('btn');
                const status = document.getElementById('status');
                const player = document.getElementById('player');

                if (!text) return alert('कृपया हिंदी में कुछ लिखें!');
                btn.disabled = true;
                status.innerText = '⏳ आवाज़ बनाई जा रही है...';

                try {{
                    const res = await fetch('/api/tts', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ text: text, length_scale: speed }})
                    }});
                    if (!res.ok) {{
                        const err = await res.json();
                        throw new Error(err.detail || 'Server Error');
                    }}
                    const blob = await res.blob();
                    player.src = URL.createObjectURL(blob);
                    player.style.display = 'block';
                    player.play();
                    status.innerText = '✅ आवाज़ तैयार है!';
                }} catch (e) {{
                    status.innerText = '🚨 ' + e.message;
                }} finally {{
                    btn.disabled = false;
                }}
            }}

            async function uploadZip() {{
                const fileInput = document.getElementById('zipFile');
                const upBtn = document.getElementById('upBtn');
                const status = document.getElementById('status');
                if (!fileInput.files.length) return alert('कृपया पहले PC से papa_voice_MORNING_READY.zip फ़ाइल चुनें!');

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                upBtn.disabled = true;
                status.innerText = '⏳ PC से ZIP अपलोड और सेटअप हो रही है (15-30 सेकंड)...';
                try {{
                    const res = await fetch('/api/upload-model', {{
                        method: 'POST',
                        body: formData
                    }});
                    const data = await res.json();
                    if (!res.ok) throw new Error(data.detail || 'Upload failed');
                    document.getElementById('badge').innerText = '✅ Model Ready (' + data.active_source + ')';
                    document.getElementById('badge').style.color = '#a6e3a1';
                    status.innerText = '🎉 Model सफलतापूर्वक लोड हो गया! अब ऊपर बटन दबाकर आवाज़ सुनें।';
                }} catch (e) {{
                    status.innerText = '🚨 Upload Error: ' + e.message;
                }} finally {{
                    upBtn.disabled = false;
                }}
            }}
        </script>
    </body>
    </html>
    """
