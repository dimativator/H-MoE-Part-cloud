import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sign_controller", ROOT / "scripts/cloud/sign_ademamix_controller.py")
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class CampaignControllerTest(unittest.TestCase):
    def test_two_queues_then_best_lr_two_gpu_continuation(self):
        candidates = {
            "queue_A": [{"lr": "2e-3", "status": "verified", "final_val_loss": 3.2},
                        {"lr": "5e-4", "status": "verified", "final_val_loss": 2.9}],
            "queue_B": [{"lr": "1e-3", "status": "diverged"},
                        {"lr": "1e-4", "status": "verified", "final_val_loss": 3.0}],
        }
        calls = []

        def submit(path, state, key, gpus, arguments):
            calls.append((key, gpus, arguments))
            state[key] = {"job": key, "results": candidates.get(key, [])}
            controller.save(path, state)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            path.write_text(json.dumps({"smoke_1gpu_verified": True}))
            with mock.patch("sys.argv", ["controller", "--state", str(path)]), \
                 mock.patch.object(controller, "submit", side_effect=submit), \
                 mock.patch.object(controller, "completed", return_value=True):
                controller.main()
            state = json.loads(path.read_text())
        self.assertEqual(calls, [("queue_A", 1, "tune --queue A"),
                                 ("queue_B", 1, "tune --queue B"),
                                 ("smoke_2gpu", 2, "smoke --gpus 2"),
                                 ("long", 2, "long --lr 5e-4")])
        self.assertEqual(state["best_lr"], "5e-4")
        self.assertEqual(state["status"], "complete")

    def test_uncertain_submission_is_never_repeated(self):
        with mock.patch.object(controller, "execute") as execute:
            with self.assertRaises(RuntimeError):
                controller.submit(Path("/unused"), {"submission_pending": {"key": "queue_A"}},
                                  "queue_A", 1, "tune --queue A")
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
