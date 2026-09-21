import torch

from data.fineweb import FineWebLiveShardedTrainReader


class _SourceReader:
    def __init__(self, rank):
        self.rank = rank
        self.batch_size = 4
        self.sequence_length = 3
        self.step = 7

    def sample_batch(self):
        values = torch.arange(12).reshape(4, 3) + self.step * 100
        self.step += 1
        return values, values + 1

    def state_dict(self):
        return {"step": self.step}

    def load_state_dict(self, state):
        self.step = state["step"]


def test_live_sharded_reader_splits_one_source_batch():
    left = FineWebLiveShardedTrainReader(
        _SourceReader(rank=0),
        physical_rank=0,
        physical_world_size=4,
        source_world_size=2,
    )
    right = FineWebLiveShardedTrainReader(
        _SourceReader(rank=0),
        physical_rank=1,
        physical_world_size=4,
        source_world_size=2,
    )

    left_x, _ = left.sample_batch()
    right_x, _ = right.sample_batch()

    assert left_x.shape == right_x.shape == (2, 3)
    assert torch.equal(torch.cat((left_x, right_x)), torch.arange(12).reshape(4, 3) + 700)
    assert left.step == right.step == 8


def test_live_sharded_reader_round_trips_state():
    reader = FineWebLiveShardedTrainReader(
        _SourceReader(rank=1),
        physical_rank=3,
        physical_world_size=4,
        source_world_size=2,
    )
    reader.sample_batch()
    state = reader.state_dict()

    restored = FineWebLiveShardedTrainReader(
        _SourceReader(rank=1),
        physical_rank=3,
        physical_world_size=4,
        source_world_size=2,
    )
    restored.load_state_dict(state)

    assert restored.step == 8
    assert restored.source_reader.step == 8
