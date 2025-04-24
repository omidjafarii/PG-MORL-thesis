"""
Script to test a policy from the PG-MORL archive with different weight preferences
"""

import pickle
import jax
import jax.numpy as jnp
import numpy as np
from pgmorl_trainer import ActorCritic, make_morl_env
from morl_hopper import MorlHopper

# Make sure your working directory is correct for imports
import sys
sys.path.append('/content/drive/MyDrive/thesis')

def test_policy(archive_path='pgmorl_archive.pkl', weights=None):
    """Test a policy from the archive with specified weights."""
    
    # Load the saved policy archive
    with open(archive_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract policies and returns from the archive
    policies = data['params']
    returns = data['returns']
    
    # Check if the archive is empty
    if len(policies) == 0:
        print("Archive is empty. No policies to test.")
        return
    
    # If no specific weights are provided, use balanced weights
    if weights is None:
        weights = jnp.array([1/3, 1/3, 1/3])  # Balanced weights for all objectives
    else:
        weights = jnp.array(weights)
        # Normalize the weights to ensure they sum to 1
        weights = weights / jnp.sum(weights)
    
    print(f"Testing with weights: {weights}")
    
    # Scalarize the returns of all policies using the given weights
    scalarized_returns = [jnp.dot(weights, r) for r in returns]
    
    # Find the best policy by selecting the one with the highest scalarized return
    best_idx = jnp.argmax(jnp.array(scalarized_returns))
    best_policy = policies[best_idx][0]
    best_returns = returns[best_idx]
    
    print(f"Selected policy {best_idx} with expected returns: {best_returns}")
    
    # Create the environment for testing using the best policy selected
    env = make_morl_env(MorlHopper, batch_size=1, episode_length=1000)
    env.weights = weights  # Set the environment weights based on the input preferences
    
    # Initialize the ActorCritic model
    network = ActorCritic(action_dim=env.action_size)
    
    # Set the random key for environment resets
    key = jax.random.PRNGKey(0)
    state = env.reset(key)  # Reset the environment to its initial state
    
    total_reward = 0  # Variable to accumulate total reward
    reward_vector = jnp.zeros(3)  # Initialize reward vector for 3 objectives
    
    # Run one episode of the environment
    for step in range(1000):
        # Get the deterministic action from the policy (using the mean action from the policy)
        mean, _, _ = network.apply(best_policy, state.obs)
        action = jnp.clip(mean, -1.0, 1.0)  # Clip actions to valid range
        
        # Step the environment using the selected action
        state = env.step(state, action)
        
        # Accumulate total reward and the reward vector for each objective
        total_reward += state.reward[0]
        reward_vector += state.metrics['reward_vector'][0]
        
        # Stop if the environment signals that the episode is done
        if state.done[0]:
            break
    
    # Print out the results after the episode completes
    print(f"Episode completed in {step+1} steps")
    print(f"Total reward: {total_reward}")
    print(f"Reward vector: {reward_vector}")
    
    return reward_vector

if __name__ == "__main__":
    # Test the policy with different weight preferences and see the impact on performance
    
    print("Testing policy with focus on forward velocity")
    test_policy(weights=[0.8, 0.1, 0.1])  # Focus on forward velocity

    print("\nTesting policy with focus on control cost")
    test_policy(weights=[0.1, 0.8, 0.1])  # Focus on control cost

    print("\nTesting policy with focus on alive bonus")
    test_policy(weights=[0.1, 0.1, 0.8])  # Focus on alive bonus
    
    print("\nTesting policy with balanced weights")
    test_policy(weights=[1/3, 1/3, 1/3])  # Balanced weights for all objectives
