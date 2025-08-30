from dataclasses import dataclass


@dataclass
class Parameters:
    """
    This class is used to encapsulate all the hyperparameters required for training
    and running the reinforcement learning agent.

    learning_rate: Learning rate for the optimizer.
    discount_factor: Discount factor for future rewards (gamma). Higher means future rewards are more important than current rewards.
    epsilon_max: Initial value for epsilon in the epsilon-greedy policy.
    epsilon_min: Minimum value for epsilon in the epsilon-greedy policy.
    epsilon_decay_rate: Rate at which epsilon decays to its minimum value.
    target_update_rate: Rate at which the target network is updated (in num episodes).
    batch_size: Size of batches sampled from the replay memory.
    replay_memory_size: Maximum size of the replay memory.
    """

    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    epsilon_max: float = 0.9
    epsilon_min: float = 0.01
    epsilon_decay_rate: int = 2500
    target_update_rate: int = 20
    batch_size: int = 128
    replay_memory_size: int = 10000
