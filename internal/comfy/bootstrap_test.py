"""Offline regression for the remote download manager's slow status responses."""
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("bootstrap", Path(__file__).with_name("bootstrap.py"))
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class DownloadStatusTest(unittest.TestCase):
    def test_slow_status_does_not_create_another_download(self):
        self.run_bootstrap(failures=1, completes=True)

    def test_unresponsive_manager_has_bounded_retries(self):
        self.run_bootstrap(failures=5, completes=False)

    def run_bootstrap(self, failures, completes):
        calls = []
        polls = 0

        def response(request, timeout):
            nonlocal polls
            method = json.loads(request.data)["method"]
            calls.append(method)
            if method == "aria2.getVersion":
                result = {}
            elif method == "aria2.addUri":
                result = "sample-gid"
            elif method == "aria2.tellStatus":
                polls += 1
                if polls <= failures:
                    raise TimeoutError("slow status response")
                result = {"status": "complete", "completedLength": "12", "totalLength": "12"}
            else:
                self.fail(method)
            return io.BytesIO(json.dumps({"result": result}).encode())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models.json").write_text("[]")
            with (patch.object(bootstrap, "ROOT", root),
                  patch.object(bootstrap, "run"),
                  patch.object(bootstrap.subprocess, "Popen", return_value=Mock(poll=lambda: None)),
                  patch.object(bootstrap.urllib.request, "urlopen", side_effect=response),
                  patch.object(bootstrap.time, "sleep"),
                  patch.object(bootstrap.os, "chdir"),
                  patch.object(bootstrap.os, "execvp") as launch):
                if completes:
                    bootstrap.main()
                    launch.assert_called_once()
                else:
                    with self.assertRaises(TimeoutError):
                        bootstrap.main()
                    launch.assert_not_called()
        self.assertEqual(calls.count("aria2.addUri"), 1)
        self.assertEqual(polls, failures + int(completes))


if __name__ == "__main__":
    unittest.main()
