"""Offline request-boundary tests; requires the MiniMax script dependencies."""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import OpenAI

HOOKS = Path(__file__).resolve().parents[1] / ".claude" / "hooks"


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MiniMaxTests(unittest.TestCase):
    def setUp(self):
        self.contexts = contextlib.ExitStack()
        self.addCleanup(self.contexts.close)
        self.contexts.enter_context(patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}, clear=True))
        self.contexts.enter_context(patch("dotenv.load_dotenv"))
        self.provider = load_script("minimax_provider", HOOKS / "utils/llm/minimax.py")
        self.requests = []
        transport = httpx.MockTransport(self.respond)
        client = self.contexts.enter_context(httpx.Client(transport=transport))
        self.contexts.enter_context(patch("openai.OpenAI", side_effect=lambda **kwargs: OpenAI(
            **kwargs, http_client=client, max_retries=0
        )))

    def respond(self, request):
        self.requests.append(request)
        return httpx.Response(200, json={
            "id": "offline-completion", "object": "chat.completion", "created": 0,
            "model": json.loads(request.content)["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "All done!"}}],
        })

    def assert_request(self, model, url, content="Hello"):
        self.assertEqual(self.provider.prompt_llm(content), "All done!")
        request = self.requests[-1]
        self.assertEqual(request.method, "POST")
        self.assertEqual(str(request.url), url)
        self.assertEqual(json.loads(request.content), {
            "model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": 100, "temperature": 0.7,
        })

    def test_default_catalog_and_request(self):
        self.assertEqual(self.provider.SUPPORTED_MODELS, ["MiniMax-M3.1-Flash", "MiniMax-M3"])
        self.assert_request("MiniMax-M3.1-Flash", "https://api.minimax.io/v1/chat/completions")

    def test_regional_requests(self):
        for region, host in [("global_en", "api.minimax.io"), ("cn_zh", "api.minimaxi.com"),
                             ("unknown", "api.minimax.io")]:
            with self.subTest(region=region), patch.dict(os.environ, {"MINIMAX_REGION": region}):
                self.assert_request("MiniMax-M3.1-Flash", f"https://{host}/v1/chat/completions")

    def test_explicit_models_are_preserved(self):
        for model in ["MiniMax-M3", "MiniMax-M2.7", "saved-custom-model"]:
            with self.subTest(model=model), patch.dict(os.environ, {"MINIMAX_MODEL": model}):
                self.assert_request(model, "https://api.minimax.io/v1/chat/completions")

    def test_custom_endpoint_takes_precedence(self):
        with patch.dict(os.environ, {"MINIMAX_REGION": "cn_zh",
                                    "MINIMAX_BASE_URL": " https://custom.example/v1/ ",
                                    "MINIMAX_MODEL": " custom-model "}):
            self.assert_request("custom-model", "https://custom.example/v1/chat/completions")

    def test_blank_model_uses_default(self):
        with patch.dict(os.environ, {"MINIMAX_MODEL": " ", "MINIMAX_BASE_URL": " "}):
            self.assert_request("MiniMax-M3.1-Flash", "https://api.minimax.io/v1/chat/completions")

    def test_message_content_is_preserved(self):
        content = [{"type": "text", "text": "Describe this"},
                   {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}}]
        self.assert_request("MiniMax-M3.1-Flash", "https://api.minimax.io/v1/chat/completions", content)

    def test_missing_key_does_not_send(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(self.provider.prompt_llm("Hello"))
        self.assertEqual(self.requests, [])

    def test_stop_hook_completion_reaches_transport(self):
        stop = load_script("stop_hook", HOOKS / "stop.py")

        def run_provider(command, **kwargs):
            self.assertEqual(command, ["uv", "run", str(HOOKS / "utils/llm/minimax.py"), "--completion"])
            stdout = io.StringIO()
            with patch("sys.argv", [command[2], command[3]]), contextlib.redirect_stdout(stdout):
                self.provider.main()
            return subprocess.CompletedProcess(command, 0, stdout.getvalue(), "")

        with patch.object(stop.subprocess, "run", side_effect=run_provider) as run:
            self.assertEqual(stop.get_llm_completion_message(), "All done!")
        run.assert_called_once()
        self.assertEqual(len(self.requests), 1)
        payload = json.loads(self.requests[0].content)
        self.assertEqual(payload["model"], "MiniMax-M3.1-Flash")
        self.assertIn("Generate a short, friendly completion message", payload["messages"][0]["content"])
        for forbidden in ["reasoning_effort", "thinking", "service_tier", "reasoning_split"]:
            self.assertNotIn(forbidden, payload)


if __name__ == "__main__":
    unittest.main()
