from train_osediff_focus_fusion import simulate_optimizer_update_schedule


def test_accumulation_one_counts_updates():
    s = simulate_optimizer_update_schedule(5, 1, 3, checkpointing_steps=1, validation_steps=2)
    assert s["global_step"] == 3
    assert s["micro_batches"] == 3
    assert s["optimizer_steps"] == 3
    assert s["scheduler_steps"] == 3
    assert s["checkpoints"] == [1, 2, 3]
    assert s["validations"] == [2]


def test_accumulation_four_max_steps_three():
    s = simulate_optimizer_update_schedule(20, 4, 3, checkpointing_steps=2, validation_steps=3)
    assert s["global_step"] == 3
    assert s["micro_batches"] == 12
    assert s["optimizer_steps"] == 3
    assert s["scheduler_steps"] == 3
    assert s["checkpoints"] == [2]
    assert s["validations"] == [3]


def test_resume_step_counts_only_remaining_updates():
    s = simulate_optimizer_update_schedule(20, 4, 3, start_global_step=2)
    assert s["global_step"] == 3
    assert s["micro_batches"] == 4
    assert s["optimizer_steps"] == 1
    assert s["scheduler_steps"] == 1
