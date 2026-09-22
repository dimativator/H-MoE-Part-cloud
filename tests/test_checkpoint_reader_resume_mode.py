from scripts.cloud.checkpoint_reader_resume_mode import contains_live_reader_state


def test_detects_nested_live_reader_state():
    state = {
        "reader_type": "fineweb_replay_train_reader_v1",
        "source_states": [
            {
                "reader_type": "fineweb_live_sharded_train_reader_v1",
                "step": 123,
            }
        ],
    }
    assert contains_live_reader_state(state)


def test_rejects_packed_reader_state():
    state = {
        "reader_type": "fineweb_replay_train_reader_v1",
        "source_states": [
            {"reader_type": "fineweb_packed_train_reader_v1", "step": 123}
        ],
    }
    assert not contains_live_reader_state(state)
