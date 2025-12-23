from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, Iterable, List


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

    def _request(self, method: str, path: str, payload: bytes | None = None) -> Any:
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        request = urllib.request.Request(url, data=payload, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                return json.load(response)
            return response.read()

    def queue_prompt(self, prompt: Dict[str, Any]) -> str:
        body = json.dumps({"prompt": prompt, "client_id": self.client_id}).encode("utf-8")
        response = self._request("POST", "/prompt", body)
        return response["prompt_id"]

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/history/{prompt_id}")

    def wait_for_prompt(
        self,
        prompt_id: str,
        *,
        poll_interval: float = 1.0,
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            history = self.get_history(prompt_id)
            entry = history.get(prompt_id)
            if entry and entry.get("outputs"):
                return history
            time.sleep(poll_interval)
        raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")

    def extract_images(self, history: Dict[str, Any], prompt_id: str) -> List[Dict[str, Any]]:
        outputs = history.get(prompt_id, {}).get("outputs", {})
        images: List[Dict[str, Any]] = []
        for node_output in outputs.values():
            images.extend(node_output.get("images", []))
        return images

    def download_image(self, filename: str, *, subfolder: str = "", image_type: str = "output") -> bytes:
        query = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": image_type}
        )
        return self._request("GET", f"/view?{query}")
