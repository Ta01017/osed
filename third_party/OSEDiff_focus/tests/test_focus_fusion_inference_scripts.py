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


def test_python_inference_uses_shared_latents_and_streaming_mae():
    source = (ROOT / "test_osediff_focus_fusion.py").read_text()
    assert "prepare_condition_latents" in source
    assert "forward_from_latents" in source
    assert "predictions[mode].append" not in source
    assert "sum_equal" in source and "sample_count" in source


def test_benchmark_uses_verified_checkpoint_and_no_zero_prompt():
    source = (ROOT / "test_focus_fusion_inference_time.py").read_text()
    assert "load_verified_checkpoint" in source
    assert "prompt = torch.zeros" not in source
    assert "prompt_embedding_source" in source


def test_smoke_shells_have_automatic_checks_and_independent_ranks():
    one = (ROOT / "scripts" / "0720_smoke_focus_fusion_1step.sh").read_text()
    resume = (ROOT / "scripts" / "0720_smoke_focus_fusion_resume.sh").read_text()
    assert "[SMOKE PASS]" in one
    assert "checkpoint_complete.json" in one
    assert "compute_file_sha256" in one
    assert "LORA_RANK_UNET" in resume
    assert "LORA_RANK_VAE" in resume
    assert "LORA_RANK_VSD" in resume
    assert "TRAIN_VAE_LORA" in resume
    assert "USE_VSD" in resume
    assert "[SMOKE PASS]" in resume
