import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _script_command(input_mode):
    env = os.environ.copy()
    env.update(
        {
            "CHECKPOINT": "/tmp/focus_fake.pt",
            "INPUT_MODE": input_mode,
            "MAX_SAMPLES": "1",
            "GPU": "",
        }
    )
    script = ROOT / "scripts" / "0720_infer_focus_fusion.sh"
    result = subprocess.run(["bash", "-x", str(script)], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.stdout


def test_single_inference_script_does_not_request_all_ablations():
    output = _script_command("single")
    assert "--condition_ablation normal" in output
    assert "--run_all_ablations" not in output


def test_dual_inference_script_defaults_to_all_ablations():
    output = _script_command("dual")
    assert "--run_all_ablations" in output
