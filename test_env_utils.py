import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import env_utils
from langchain_openai import ChatOpenAI


class GatewaySetupTests(unittest.TestCase):
    def test_required_keys_follow_model_route(self):
        required = {
            "OPENAI_API_KEY": "your-openai-api-key",
            "LANGSMITH_API_KEY": "lsv2_...",
        }
        for flag in (
            "",
            "false",
            "0",
            "no",
            "FALSE",
            "true",
            "1",
            "yes",
            "https://eu.gateway.smith.langchain.com",
        ):
            for override in ("", "gateway-key"):
                with self.subTest(flag=flag, override=override):
                    environ = {
                        "LANGSMITH_GATEWAY": flag,
                        "LANGSMITH_GATEWAY_API_KEY": override,
                    }
                    selected = env_utils._required_keys_for_mode(required, environ)
                    enabled = flag.lower() not in ("", "false", "0", "no")
                    self.assertEqual("OPENAI_API_KEY" in selected, not enabled)
                    self.assertEqual(
                        "LANGSMITH_GATEWAY_API_KEY" in selected,
                        enabled and bool(override),
                    )
                    self.assertIn("LANGSMITH_API_KEY", selected)
        selected = env_utils._required_keys_for_mode(
            required,
            {
                "LANGSMITH_GATEWAY": "true",
                "OPENAI_API_BASE": "https://api.openai.com/v1",
            },
        )
        self.assertIn("OPENAI_API_KEY", selected)

    def test_reports_missing_empty_and_placeholder_gateway_key(self):
        for value, expected in (
            (None, "is required but not set"),
            ("", "is empty"),
            ("lsv2_...", "still has the example/placeholder value"),
        ):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"LANGSMITH_GATEWAY": "true"}, clear=True),
            ):
                if value is not None:
                    os.environ["LANGSMITH_API_KEY"] = value
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    env_utils.doublecheck_env(
                        str(Path(__file__).with_name(".env.example"))
                    )
                self.assertIn("LANGSMITH_API_KEY " + expected, output.getvalue())
                self.assertNotIn("OPENAI_API_KEY is required", output.getvalue())

    def test_shell_only_does_not_accept_empty_values(self):
        with tempfile.TemporaryDirectory() as directory:
            example = Path(directory, ".env.example")
            example.write_text("OPENAI_API_KEY=your-openai-api-key\n")
            with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    env_utils.check_env_file_exists(
                        str(Path(directory, ".env")), str(example)
                    )
                self.assertIn("Required variables not set", output.getvalue())

    def test_checker_loads_the_file_it_checked_over_shell_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath(".env.example").write_text(
                "OPENAI_API_KEY=your-openai-api-key\n"
            )
            root.joinpath(".env").write_text(
                "LANGSMITH_GATEWAY=false\nOPENAI_API_KEY=\n"
            )
            result = subprocess.run(
                [sys.executable, str(Path(env_utils.__file__).resolve())],
                cwd=root,
                env={
                    **os.environ,
                    "LANGSMITH_GATEWAY": "true",
                    "OPENAI_API_KEY": "shell-provider-key",
                },
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("OPENAI_API_KEY is empty", result.stdout)
            self.assertIn(".env values take precedence", result.stdout)
            self.assertNotIn("shell-provider-key", result.stdout)

    def test_chat_model_uses_selected_credentials_without_requests(self):
        for flag, host, key in (
            ("", "https://api.openai.com/v1/", "provider-key"),
            ("false", "https://api.openai.com/v1/", "provider-key"),
            ("true", "https://gateway.smith.langchain.com/openai/v1/", "langsmith-key"),
            (
                "https://eu.gateway.smith.langchain.com",
                "https://eu.gateway.smith.langchain.com/openai/v1/",
                "langsmith-key",
            ),
        ):
            for override in ("", "gateway-key"):
                with (
                    self.subTest(flag=flag, override=override),
                    patch.dict(
                        os.environ,
                        {
                            "LANGSMITH_GATEWAY": flag,
                            "OPENAI_API_KEY": "provider-key",
                            "LANGSMITH_API_KEY": "langsmith-key",
                            "LANGSMITH_GATEWAY_API_KEY": override,
                            "LANGSMITH_TRACING": "false",
                        },
                        clear=True,
                    ),
                ):
                    model = ChatOpenAI(model="gpt-4o-mini")
                    self.assertEqual(str(model.root_client.base_url), host)
                    self.assertEqual(
                        model.root_client.api_key,
                        override if override and flag not in ("", "false") else key,
                    )


if __name__ == "__main__":
    unittest.main()
