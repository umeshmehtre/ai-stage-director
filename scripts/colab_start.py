"""One command to start everything on Colab: GPU libraries + server + public tunnel.

Run from anywhere:
    python scripts/colab_start.py --model /content/ai-stage-director/models/qwen-stage-director
"""

import argparse
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
)


def gpu_lib_path() -> str:
    """Folders holding libcublas/libcudnn (needed by Whisper on GPU)."""
    try:
        import nvidia.cublas
        import nvidia.cudnn
    except ImportError:
        print("WARNING: nvidia libs missing. Run: pip install -e '.[gpu]'")
        return ""
    return ":".join(str(Path(m.__path__[0]) / "lib") for m in (nvidia.cublas, nvidia.cudnn))


def wait_for_health(port: int, seconds: int = 60) -> bool:
    for _ in range(seconds):
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2) as r:
                return r.status == 200
        except Exception:
            time.sleep(1)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/qwen-stage-director")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--language", default="en", help="Whisper language ('' = auto)")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--no-tts", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    model = str(Path(args.model).resolve())
    if not Path(model).exists():
        raise SystemExit(f"Model folder not found: {model}")

    subprocess.run("pkill -f uvicorn; pkill -f cloudflared", shell=True)
    time.sleep(2)

    env = {
        **os.environ,
        "LD_LIBRARY_PATH": gpu_lib_path() + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
        "STAGE_DIRECTOR_BACKEND": "llm",
        "STAGE_LLM_MODEL_ID": model,
        "STAGE_WHISPER_LANGUAGE": args.language,
        "STAGE_WHISPER_TASK": args.task,
        "STAGE_TTS_BACKEND": "none" if args.no_tts else "kokoro",
    }
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "stage_director.main:app",
         "--host", "0.0.0.0", "--port", str(args.port)],
        env=env, stdout=open("server.log", "w"), stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print("Server started:", "OK" if wait_for_health(args.port) else "FAILED (see server.log)")

    cloudflared = Path("cloudflared-linux-amd64")
    if not cloudflared.exists():
        subprocess.run(["wget", "-q", CLOUDFLARED_URL], check=True)
        cloudflared.chmod(0o755)
    subprocess.Popen(
        [f"./{cloudflared}", "tunnel", "--url", f"http://localhost:{args.port}"],
        stdout=open("tunnel.log", "w"), stderr=subprocess.STDOUT, start_new_session=True,
    )

    for _ in range(30):
        time.sleep(1)
        match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", Path("tunnel.log").read_text())
        if match:
            print(f"\n>>> API URL: {match.group(0)}")
            return
    print("Tunnel failed. Run: cat tunnel.log")


if __name__ == "__main__":
    main()