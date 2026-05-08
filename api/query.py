import json
import os
import shutil
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.chdir(REPO_ROOT)

SRC_CHROMA = REPO_ROOT / "chroma"
TMP_CHROMA = Path("/tmp/chroma")

_state = {"ready": False, "ensemble": None, "client": None, "ranker": None, "error": None}


def _initialize():
    if _state["ready"]:
        return
    try:
        if SRC_CHROMA.exists() and not TMP_CHROMA.exists():
            shutil.copytree(SRC_CHROMA, TMP_CHROMA)

        import src.pipeline as pipeline
        pipeline.CHROMA_PATH = str(TMP_CHROMA)

        from openai import OpenAI
        from src.pipeline import (
            get_ensemble_retriever,
            get_reranker_model,
            load_documents,
        )

        documents = load_documents()
        ensemble = get_ensemble_retriever(documents)
        ranker = get_reranker_model()
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_KEY"),
        )

        _state.update(ensemble=ensemble, client=client, ranker=ranker, ready=True)
    except Exception as e:
        _state["error"] = f"{type(e).__name__}: {e}"
        raise


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(204, {})

    def do_GET(self):
        try:
            _initialize()
            self._send_json(200, {"status": "ok"})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return

        query_text = (payload.get("query") or "").strip()
        if not query_text:
            self._send_json(400, {"error": "query is required"})
            return

        try:
            _initialize()
            from src.query import query_rag
            response = query_rag(
                query_text,
                _state["ensemble"],
                _state["client"],
                _state["ranker"],
            )
            self._send_json(200, {"response": response})
        except Exception as e:
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})
