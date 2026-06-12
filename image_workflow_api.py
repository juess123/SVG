from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_NOTIFY_URL = "http://192.168.2.104/imageWorkflow/notify"
DEFAULT_OUTPUT_COS_PREFIX = "svg-output"
FINAL_SVG_NAME = "layout.svg"
DEFAULT_STATUS_PATH = SCRIPT_DIR / "workflow_jobs.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def load_env_file(path: Path = SCRIPT_DIR / ".env") -> None:
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {remaining:.1f}s"


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or f"job_{int(time.time())}"


def image_suffix(cos_key: str) -> str:
    suffix = Path(cos_key.split("?", 1)[0]).suffix.lower()
    return suffix if suffix in IMAGE_EXTENSIONS else ".png"


def normalize_cos_key(key: str) -> str:
    return key.strip().lstrip("/")


def json_response(code: int, data: dict[str, Any] | None, msg: str) -> dict[str, Any]:
    return {"code": code, "data": data or {}, "msg": msg}


class SubmitRequest(BaseModel):
    image_cos_key: str
    order_id: str


class CosStorage:
    def __init__(self) -> None:
        try:
            from qcloud_cos import CosConfig, CosS3Client
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: cos-python-sdk-v5. Install dependencies with "
                "`venv\\Scripts\\python.exe -m pip install -r requirements.txt`."
            ) from exc

        secret_id = os.environ.get("COS_SECRET_ID")
        secret_key = os.environ.get("COS_SECRET_KEY")
        region = os.environ.get("COS_REGION")
        bucket = os.environ.get("COS_BUCKET")
        if not secret_id or not secret_key or not region or not bucket:
            raise RuntimeError("COS config missing. Set COS_SECRET_ID, COS_SECRET_KEY, COS_REGION, and COS_BUCKET.")

        config = CosConfig(
            Region=region,
            SecretId=secret_id,
            SecretKey=secret_key,
            Token=os.environ.get("COS_TOKEN"),
            Scheme=os.environ.get("COS_SCHEME", "https"),
        )
        self.client = CosS3Client(config)
        self.bucket = bucket

    def download_file(self, cos_key: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        response = self.client.get_object(Bucket=self.bucket, Key=normalize_cos_key(cos_key))
        response["Body"].get_stream_to_file(str(local_path))
        return local_path

    def upload_file(self, local_path: Path, cos_key: str) -> str:
        content_type = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
        self.client.upload_file(
            Bucket=self.bucket,
            LocalFilePath=str(local_path),
            Key=normalize_cos_key(cos_key),
            PartSize=10,
            MAXThread=4,
            EnableMD5=False,
            ContentType=content_type,
        )
        return normalize_cos_key(cos_key)


class JobStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def update(self, order_id: str, **fields: Any) -> dict[str, Any]:
        with self.lock:
            data = self._read()
            current = data.get(order_id, {})
            current.update(fields)
            current["updated_at"] = now_iso()
            data[order_id] = current
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return current

    def get(self, order_id: str) -> dict[str, Any] | None:
        with self.lock:
            return self._read().get(order_id)


job_store = JobStore(Path(os.environ.get("WORKFLOW_STATUS_PATH", str(DEFAULT_STATUS_PATH))).resolve())
app = FastAPI(title="SVG Image Workflow API")


def notify_result(order_id: str, svg_cos_key: str, status: str) -> None:
    notify_url = os.environ.get("WORKFLOW_NOTIFY_URL", DEFAULT_NOTIFY_URL)
    payload = {
        "order_id": order_id,
        "svg_cos_key": svg_cos_key,
        "status": status,
    }
    timeout = float(os.environ.get("WORKFLOW_NOTIFY_TIMEOUT", "30"))
    response = requests.post(notify_url, json=payload, timeout=timeout)
    response.raise_for_status()


def run_pipeline_for_image(image_path: Path) -> Path:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_batch_pipeline.py"),
        "-i",
        str(image_path),
        "--force-api",
        "--force-relayout",
    ]
    print(f"Running workflow command: {' '.join(command)}")
    completed = subprocess.run(
        command,
        cwd=SCRIPT_DIR,
        text=True,
        capture_output=True,
        timeout=int(os.environ.get("WORKFLOW_PIPELINE_TIMEOUT", "3600")),
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"Pipeline failed with exit code {completed.returncode}")

    output_path = SCRIPT_DIR / "output" / image_path.stem / "out" / FINAL_SVG_NAME
    if not output_path.is_file():
        raise FileNotFoundError(f"Final SVG not found: {output_path}")
    return output_path


def process_job(order_id: str, image_cos_key: str, image_path: Path) -> None:
    started = time.perf_counter()
    svg_cos_key = ""
    try:
        job_store.update(order_id, status="processing", started_at=now_iso())
        svg_path = run_pipeline_for_image(image_path)

        output_prefix = normalize_cos_key(os.environ.get("SVG_OUTPUT_COS_PREFIX", DEFAULT_OUTPUT_COS_PREFIX))
        svg_cos_key = f"{output_prefix}/{safe_name(order_id)}/{FINAL_SVG_NAME}"
        storage = CosStorage()
        storage.upload_file(svg_path, svg_cos_key)

        job_store.update(
            order_id,
            status="success",
            image_cos_key=image_cos_key,
            svg_cos_key=svg_cos_key,
            svg_path=str(svg_path),
            duration=format_duration(time.perf_counter() - started),
            finished_at=now_iso(),
        )
        notify_result(order_id, svg_cos_key, "success")
    except Exception as exc:
        error = f"{exc}"
        print(f"Workflow failed for order_id={order_id}: {error}", file=sys.stderr)
        traceback.print_exc()
        job_store.update(
            order_id,
            status="fail",
            image_cos_key=image_cos_key,
            svg_cos_key=svg_cos_key,
            error=error,
            duration=format_duration(time.perf_counter() - started),
            finished_at=now_iso(),
        )
        try:
            notify_result(order_id, svg_cos_key, "fail")
        except Exception as notify_exc:
            print(f"Notify failed for order_id={order_id}: {notify_exc}", file=sys.stderr)
            job_store.update(order_id, notify_error=str(notify_exc))


@app.get("/health")
def health() -> dict[str, Any]:
    return json_response(200, {"status": "ok"}, "ok")


@app.get("/imageWorkflow/jobs/{order_id}")
def job_status(order_id: str) -> dict[str, Any]:
    item = job_store.get(order_id)
    if not item:
        return json_response(404, {}, "任务不存在")
    return json_response(200, item, "ok")


@app.post("/imageWorkflow/submit")
def submit_workflow(request: SubmitRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    image_cos_key = request.image_cos_key.strip()
    order_id = request.order_id.strip()
    if not image_cos_key:
        return json_response(400, {}, "image_cos_key不能为空")
    if not order_id:
        return json_response(400, {}, "order_id不能为空")

    local_name = safe_name(order_id) + image_suffix(image_cos_key)
    local_path = SCRIPT_DIR / "input" / local_name

    try:
        storage = CosStorage()
        storage.download_file(image_cos_key, local_path)
        job_store.update(
            order_id,
            status="submitted",
            image_cos_key=image_cos_key,
            image_path=str(local_path),
            submitted_at=now_iso(),
        )
    except Exception as exc:
        error = f"{exc}"
        print(f"Submit failed for order_id={order_id}: {error}", file=sys.stderr)
        traceback.print_exc()
        return json_response(500, {}, f"下载图片失败: {error}")

    background_tasks.add_task(process_job, order_id, image_cos_key, local_path)
    return json_response(200, {}, "任务已提交")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SVG image workflow API.")
    parser.add_argument("--host", default=os.environ.get("WORKFLOW_API_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("WORKFLOW_API_PORT", "8000")))
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("image_workflow_api:app", host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
