import json
import unittest
from unittest.mock import patch

import httpx

from app.schemas.contracts import AppSettingsSchema
from app.services.provider_service import ModelProviderService


class GeminiProviderTests(unittest.IsolatedAsyncioTestCase):
    async def run_provider(self, response, messages=None, model="gemini-3.5-flash"):
        self.requests = []

        def handler(request):
            self.requests.append(request)
            return response

        config = {
            "model_api_key": "test-secret",
            "model_api_base_url": "https://generativelanguage.googleapis.com/v1beta",
            "model_api_wire_api": "gemini",
            "model_api_headers": {"x-provider-auth": "header-secret"},
        }
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch("app.services.provider_service.storage_service.get_settings", return_value=config), \
             patch("app.services.provider_service.httpx.AsyncClient", return_value=client):
            return [event async for event in ModelProviderService().stream_chat_completion(
                model, messages or [{"role": "user", "content": "Hello"}], "Be helpful",
            )]

    @staticmethod
    def sse(*events):
        body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    async def test_stream_uses_gemini_contract(self):
        messages = [
            {"role": "assistant", "content": "Earlier answer"},
            {"role": "user", "content": "First question"},
            {"role": "user", "content": "Describe", "image_url": "data:image/png;base64,YQ=="},
        ]
        events = await self.run_provider(self.sse(
            {"candidates": [{"content": {"parts": [
                {"thought": True, "text": "hidden plan"},
                {"text": "Hello "},
            ]}}]},
            {"candidates": [{"content": {"parts": [{"text": "world"}]}, "finishReason": "STOP"}]},
        ), messages)
        self.assertEqual(events, [
            {"type": "content.delta", "delta": "Hello "},
            {"type": "content.delta", "delta": "world"},
            {"type": "turn.completed", "ok": True},
        ])
        request = self.requests[0]
        self.assertEqual(
            str(request.url),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:streamGenerateContent?alt=sse",
        )
        self.assertEqual(request.headers["x-goog-api-key"], "test-secret")
        self.assertEqual(request.headers["x-provider-auth"], "header-secret")
        body = json.loads(request.content)
        self.assertEqual(body["systemInstruction"], {"parts": [{"text": "Be helpful"}]})
        self.assertEqual(body["contents"][0]["role"], "user")
        self.assertEqual(body["contents"][0]["parts"][0], {"text": "First question"})
        self.assertEqual(body["contents"][0]["parts"][1]["text"], "Describe")
        self.assertEqual(body["contents"][0]["parts"][2]["inline_data"]["mime_type"], "image/png")
        self.assertNotIn("Earlier answer", request.content.decode())

    async def test_http_and_blocked_responses_do_not_expose_secrets(self):
        events = await self.run_provider(httpx.Response(401, text="test-secret header-secret"))
        self.assertFalse(events[-1]["ok"])
        self.assertIn("401", events[0]["delta"])
        self.assertNotIn("test-secret", json.dumps(events))
        self.assertNotIn("header-secret", json.dumps(events))

        events = await self.run_provider(self.sse({"promptFeedback": {"blockReason": "SAFETY"}, "error": {"message": "test-secret"}}))
        self.assertFalse(events[-1]["ok"])
        self.assertNotIn("test-secret", json.dumps(events))
        self.assertNotIn("SAFETY", json.dumps(events))

    async def test_invalid_model_id_is_not_requested(self):
        events = await self.run_provider(self.sse({}), model="../gemini-3.5-flash")
        self.assertFalse(events[-1]["ok"])
        self.assertEqual(self.requests, [])

    def test_settings_accept_gemini_protocol(self):
        settings = AppSettingsSchema(model_api_wire_api="gemini", default_model="gemini-3.5-flash")
        self.assertEqual(settings.model_api_wire_api, "gemini")
