from mobile_manipulation_il_rl.envs.fsm import Phase, TaskFSM


def test_phase_order_and_timeout():
    fsm = TaskFSM({"NAVIGATE_TO_PICK": 1})
    assert fsm.phase == Phase.NAVIGATE_TO_PICK
    fsm.advance("done"); assert fsm.phase == Phase.REACH
    fsm.reset(); fsm.tick(1.1)
    assert fsm.phase == Phase.FAILURE and "timeout" in fsm.failure_reason

