from brax.envs.half_cheetah import Halfcheetah as BaseHalfcheetah
from brax.v1.envs.env import State
import jax.numpy as jnp

class MorlHalfcheetah(BaseHalfcheetah):
    def __init__(self, **kwargs):
        # Initialize the parent HalfCheetah environment
        super().__init__(**kwargs)

        # Set default MORL scalarization weights: [speed, energy]
        self.weights = jnp.array([1.0, 0.0])  # Can be changed during training

    def reset(self, rng) -> State:
        # Reset the environment using the base implementation
        state = super().reset(rng)

        # Add reward_vector for MORL tracking
        reward_vector = jnp.zeros(2)
        state = state.replace(
            metrics={**state.metrics, 'reward_vector': reward_vector}
        )
        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        # Perform one environment step
        state = super().step(state, action)

        # Objective 1: Forward velocity (usually at obs[0] for HalfCheetah)
        forward_vel = state.obs[0]

        # Objective 2: Negative control cost (energy efficiency)
        ctrl_cost = jnp.sum(jnp.square(action))

        # Final reward vector [forward velocity, -energy]
        reward_vector = jnp.array([forward_vel, -ctrl_cost])

        # Scalarized reward
        scalar_reward = jnp.dot(self.weights, reward_vector)

        # Track reward vector for logging/visualization
        updated_metrics = {**state.metrics, 'reward_vector': reward_vector}

        return state.replace(
            reward=scalar_reward,
            metrics=updated_metrics
        )
