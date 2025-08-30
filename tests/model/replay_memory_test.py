from src.model.replay_memory import ReplayMemory, MemoryFragment


def make_memory_fragment(score: int):
    return MemoryFragment([], 0, [], score)


def test_memory_can_be_initialized():
    memory = ReplayMemory(capacity=2)
    assert len(memory) == 0


def test_memory_keeps_latest_elements():
    memory = ReplayMemory(capacity=2, seed=1)

    memory.push(make_memory_fragment(0))
    memory.push(make_memory_fragment(1))
    memory.push(make_memory_fragment(2))

    assert len(memory) == 2
    assert memory.sample(2) == [make_memory_fragment(1), make_memory_fragment(2)]
