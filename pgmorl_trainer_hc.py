"""
PG-MORL Trainer for Brax Environments
Based on the paper: "Learning and Visualizing Pareto Optimal Policies using PG-MORL"
Memory-optimized version
"""

import os
import time
from typing import Callable, Dict, Optional, Sequence, Tuple
import gc

import flax
import flax.linen as nn
import jax
import jax.tree_util as jtree
import jax.numpy as jnp
import numpy as np
import optax
from brax.training.types import Params, PRNGKey
from brax.envs import Env
import mywrappers
import functools

# Define actor and critic networks
class ActorCritic(nn.Module):
    action_dim: int
    hidden_layer_sizes: Sequence[int] = (256, 256)
    
    @nn.compact
    def __call__(self, x):
        # Actor network (policy): Predicts action mean and log std for exploration
        actor_mean = x
        for size in self.hidden_layer_sizes:
            actor_mean = nn.Dense(size)(actor_mean)
            actor_mean = nn.tanh(actor_mean)
        actor_mean = nn.Dense(self.action_dim)(actor_mean)
        
        # Actor logstd parameter (controls the randomness of actions)
        actor_logstd = self.param('actor_logstd', 
                                  lambda _: -0.5 * jnp.ones(self.action_dim))
        
        # Critic network (value function): Estimates the value of each state
        critic = x
        for size in self.hidden_layer_sizes:
            critic = nn.Dense(size)(critic)
            critic = nn.tanh(critic)
        critic = nn.Dense(2)(critic)  # 2-dimensional value for 2 objectives
        
        return actor_mean, actor_logstd, critic

def make_morl_env(env_name: str, batch_size: int, episode_length: int, action_repeat: int):
    """Creates a batched MORL environment."""
    env = env_name()
    # Wrap environment with necessary wrappers: batch processing, episode tracking, auto-reset
    env = mywrappers.VmapWrapper(env)
    env = mywrappers.EpisodeWrapper(env, episode_length, action_repeat)
    env = mywrappers.AutoResetWrapper(env)
    return env

def sample_weights(n_samples: int, n_objectives: int = 2) -> jnp.ndarray:
    """Sample weights from a simplex for the scalarization."""
    weights = np.random.dirichlet(np.ones(n_objectives), size=n_samples)
    return jnp.array(weights)

class PGMORLTrainer:
    """Policy Gradient Multi-Objective Reinforcement Learning Trainer."""
    
    def __init__(self, 
                 env_fn,
                 num_timesteps: int = 1_000_000,
                 episode_length: int = 1000,
                 action_repeat: int = 1,
                 num_envs: int = 128,
                 num_eval_envs: int = 128,
                 max_devices_per_host: Optional[int] = None,
                 learning_rate: float = 3e-4,
                 entropy_cost: float = 1e-2,
                 unroll_length: int = 10,
                 batch_size: int = 32,
                 num_minibatches: int = 16,
                 num_updates_per_batch: int = 4,
                 discounting: float = 0.99,
                 seed: int = 0,
                 num_objectives: int = 2):
        
        # Initialize hyperparameters and environment settings
        self.num_timesteps = num_timesteps
        self.episode_length = episode_length
        self.action_repeat = action_repeat
        self.num_envs = num_envs
        self.max_devices_per_host = max_devices_per_host
        self.learning_rate = learning_rate
        self.entropy_cost = entropy_cost
        self.unroll_length = unroll_length
        self.batch_size = batch_size
        self.num_minibatches = num_minibatches
        self.num_updates_per_batch = num_updates_per_batch
        self.discounting = discounting
        self.seed = seed
        self.num_objectives = num_objectives
        
        # Initialize environment and evaluation environments
        self.env = make_morl_env(env_fn, num_envs, episode_length, action_repeat)
        self.eval_env = make_morl_env(env_fn, num_eval_envs, episode_length, action_repeat)
        self.eval_batch_size = num_eval_envs

        # Initialize network (ActorCritic model) and optimizer
        self._init_model()
        
        # Archive for storing Pareto policies
        self.archive = ParetoArchive(num_objectives=self.num_objectives)
        
    def _init_model(self):
        """Initialize the ActorCritic model and optimizer."""
        self.key = jax.random.PRNGKey(self.seed)
        self.key, network_key, reset_key = jax.random.split(self.key, 3)
        
        # Get environment dimensions and reset the environment
        reset_key = jax.random.split(reset_key, self.num_envs)
        env_state = self.env.reset(reset_key)
        obs = env_state.obs
        
        # Initialize the ActorCritic network
        self.network = ActorCritic(action_dim=self.env.action_size)
        self.network_params = self.network.init(network_key, obs[0])
        
        # Initialize optimizer (Adam with global gradient clipping)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=self.learning_rate)
        )
        self.optimizer_state = self.optimizer.init(self.network_params)
        
    def _policy_loss(self, params, obs, actions, advantages, old_log_probs):
        """Calculate policy loss with clipped objective.
        Scalarize advantages across objectives using current weights.
        """
        mean, log_std, _ = self.network.apply(params, obs)
        log_probs = self._compute_log_prob(mean, log_std, actions)
        
        # Compute ratio and clip to avoid large updates
        ratio = jnp.exp(log_probs - old_log_probs)
        clipped_ratio = jnp.clip(ratio, 0.8, 1.2)
        
        # Scalarize advantages (weighted sum for each objective)
        scalar_advantages = jnp.dot(advantages, self.env.weights)

        # Policy loss using surrogate function
        surrogate_loss1 = ratio * scalar_advantages
        surrogate_loss2 = clipped_ratio * scalar_advantages
        policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))
        
        # Entropy bonus for exploration
        entropy = jnp.mean(jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1))
        entropy_loss = -self.entropy_cost * entropy
        
        return policy_loss + entropy_loss
    
    def _value_loss(self, params, obs, returns):
        """Calculate value loss for all objectives."""
        _, _, values = self.network.apply(params, obs)
        value_loss = jnp.mean(jnp.sum((values - returns) ** 2, axis=-1))
        return value_loss
    
    def _compute_log_prob(self, mean, log_std, actions):
        """Compute log probability of actions under the policy."""
        std = jnp.exp(log_std)
        log_prob = -0.5 * jnp.sum(
            ((actions - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
        return log_prob
    
    def _update_step(self, params, opt_state, obs, actions, advantages, returns, old_log_probs):
        """Single update step for policy and value function."""
        def loss_fn(params):
            # Compute policy and value losses
            policy_loss = self._policy_loss(params, obs, actions, advantages, old_log_probs)
            value_loss = self._value_loss(params, obs, returns)
            total_loss = policy_loss + 0.5 * value_loss
            return total_loss, (policy_loss, value_loss)
        
        # Compute loss and gradients
        (loss, (policy_loss, value_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        
        # Apply gradients using the optimizer
        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        metrics = {
            'loss': loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
        }
        
        return new_params, new_opt_state, metrics
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _sample_batch(self, key, params, env_state):
        """Collect a batch of data by rolling out the policy."""
        def step_fn(carry, _):
            key, state, params = carry
            key, key_action, key_step = jax.random.split(key, 3)
            
            # Get policy action (sample from distribution)
            mean, log_std, values = self.network.apply(params, state.obs)
            std = jnp.exp(log_std)
            action = mean + std * jax.random.normal(key_action, mean.shape)
            action = jnp.clip(action, -1.0, 1.0)
            
            # Step environment and compute log probability
            next_state = self.env.step(state, action)
            log_prob = self._compute_log_prob(mean, log_std, action)
            
            # Store data in the transition
            transition = {
                'obs': state.obs,
                'action': action,
                'reward_vector': next_state.metrics['reward_vector'],
                'reward': next_state.reward,
                'value': values,
                'log_prob': log_prob,
                'done': next_state.done,
            }
            
            return (key, next_state, params), transition
        
        # Collect trajectories
        (key, final_state, _), traj = jax.lax.scan(step_fn, (key, env_state, params), None, length=self.unroll_length)
        
        # Compute advantages and returns for each objective
        advantages = []
        returns = []
        
        # For each objective (speed, stability, energy efficiency)
        for obj_idx in range(self.num_objectives):
            obj_rewards = traj['reward_vector'][:, :, obj_idx]
            obj_values = traj['value'][:, :, obj_idx]
            advantages.append(self._compute_advantages(obj_rewards, obj_values))
            returns.append(self._compute_returns(advantages[obj_idx], obj_values))
        
        # Stack advantages and returns for all objectives
        advantages = jnp.stack(advantages, axis=-1)
        returns = jnp.stack(returns, axis=-1)
        
        batch = {
            'obs': traj['obs'],
            'action': traj['action'],
            'advantage': advantages,
            'return': returns,
            'log_prob': traj['log_prob'],
            'reward_vector': traj['reward_vector'],
        }
        
        return key, final_state, batch
    
    def _compute_advantages(self, rewards, values):
        """Compute Generalized Advantage Estimation (GAE) for each objective."""
        advantages = []
        gae = jnp.zeros((self.num_envs,))
        for t in reversed(range(self.unroll_length)):
            delta = rewards[t] + self.discounting * values[t + 1] - values[t]
            gae = delta + self.discounting * 0.95 * gae
            advantages.insert(0, gae)
        return jnp.stack(advantages)
    
    def _compute_returns(self, advantages, values):
        """Compute returns from advantages and values."""
        return advantages + values
    
    def train(self):
        """Main training loop with memory optimization."""
        print("Starting PG-MORL training...")
        
        # Initialize environment and reset
        key, reset_key = jax.random.split(self.key)
        reset_key = jax.random.split(reset_key, self.num_envs)
        env_state = self.env.reset(reset_key)
        
        # Training metrics
        metrics = {'step': 0, 'weights': []}
        start_time = time.time()
        
        # Training loop for the given number of timesteps
        num_updates = self.num_timesteps // (self.num_envs * self.unroll_length)
        print(f"Total updates to perform: {num_updates}")
        
        # Add progress counter
        last_print_time = time.time()
        
        for update in range(num_updates):
            print(f"Starting update {update+1}, collecting batch...")
            
            # Sample new weights periodically to change objective emphasis
            if update % 10 == 0:
                key, weights_key = jax.random.split(key)
                weights = sample_weights(1)[0]
                # Update environment weights for scalarization
                self.env.weights = weights
                metrics['weights'].append(weights)
            
            # Collect batch of trajectories
            key, env_state, batch = self._sample_batch(key, self.network_params, env_state)
            
            print(f"Batch collected, starting network updates...")
            
            # Prepare batch for updates
            batch_size = self.num_envs * self.unroll_length
            batch = jtree.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                
            # Normalize advantages for stable learning
            batch['advantage'] = (batch['advantage'] - batch['advantage'].mean()) / (batch['advantage'].std() + 1e-8)
            
            # Update policy and value function using mini-batches
            for _ in range(self.num_updates_per_batch):
                key, subkey = jax.random.split(key)
                permutation = jax.random.permutation(subkey, batch_size)
                
                # Mini-batch updates for policy and value function
                for start in range(0, batch_size, self.batch_size):
                    end = min(start + self.batch_size, batch_size)
                    minibatch_indices = permutation[start:end]
                    
                    minibatch = jtree.tree_map(lambda x: x[minibatch_indices], batch)
                    
                    self.network_params, self.optimizer_state, update_metrics = self._update_step(
                        self.network_params,
                        self.optimizer_state,
                        minibatch['obs'],
                        minibatch['action'],
                        minibatch['advantage'],
                        minibatch['return'],
                        minibatch['log_prob']
                    )
                    
                    metrics.update(update_metrics)
            
            # Free up memory after each update
            batch = None
            minibatch = None
            gc.collect()
            
            # Evaluate policy less frequently to save memory
            if update % 20 == 0:
                returns = self._evaluate_policy(self.network_params, num_episodes=5)
                self.archive.update(self.network_params, returns)
                
                # Log Pareto front information after evaluation
                print(f"  > Archive Size: {len(self.archive)}")
                print(f"  > Current weights: {weights}")
                print(f"  > Average returns: {returns.mean(axis=0)}")
                
                # Free memory after evaluation
                gc.collect()
        
        print("Training complete!")
        
        # Return trained parameters and Pareto archive
        return self.network_params, self.archive
    
    def _evaluate_policy(self, params, num_episodes=10):
        """Evaluate current policy and return average vector rewards."""
        # Reset evaluation environment
        key, reset_key = jax.random.split(self.key)
        self.key = key
        reset_key = jax.random.split(reset_key, self.eval_batch_size)
        eval_state = self.eval_env.reset(reset_key)
        
        # Storage for returns
        episode_returns = []
        
        # Run episodes for evaluation
        for _ in range(num_episodes):
            returns = jnp.zeros((self.eval_batch_size, self.num_objectives))
            done = jnp.zeros((self.eval_batch_size,), dtype=bool)
            
            # Run episode
            for _ in range(self.episode_length):
                # Get deterministic policy action for evaluation
                mean, _, _ = self.network.apply(params, eval_state.obs)
                action = jnp.clip(mean, -1.0, 1.0)
                
                # Step environment and accumulate rewards
                eval_state = self.eval_env.step(eval_state, action)
                reward_vector = eval_state.metrics['reward_vector']
                returns = returns + ~done[:, None] * reward_vector
                done = jnp.logical_or(done, eval_state.done)
                
                # Break if all environments are done
                if jnp.all(done):
                    break
            
            episode_returns.append(returns)
        
        # Return average rewards from evaluation episodes
        average_returns = jnp.mean(jnp.stack(episode_returns), axis=(0, 1))
        return average_returns


class ParetoArchive:
    """Archive to store Pareto optimal policies."""
    
    def __init__(self, num_objectives=2, epsilon=0.01):
        self.policies = []  # List of (params, returns) tuples
        self.returns = []   # List of return vectors
        self.num_objectives = num_objectives
        self.epsilon = epsilon  # Hypervolume contribution threshold
        
    def __len__(self):
        return len(self.policies)
    
    def update(self, params, returns):
        """Update archive with new policy if it's non-dominated."""
        if len(self.policies) == 0:
            # First policy, add it directly
            self.policies.append((params, returns))
            self.returns.append(returns)
            return True
        
        # Check if the new policy is dominated by any existing policy
        is_dominated = False
        for existing_returns in self.returns:
            if self._dominates(existing_returns, returns, tolerance=self.epsilon):
                is_dominated = True
                break
        
        if not is_dominated:
            # Remove policies that the new one dominates
            non_dominated_indices = []
            for i, existing_returns in enumerate(self.returns):
                if not self._dominates(returns, existing_returns, tolerance=0):
                    non_dominated_indices.append(i)
            
            # Keep only non-dominated policies
            self.policies = [self.policies[i] for i in non_dominated_indices]
            self.returns = [self.returns[i] for i in non_dominated_indices]
            
            # Add the new policy
            self.policies.append((params, returns))
            self.returns.append(returns)
            return True
            
        return False
    
    def _dominates(self, a, b, tolerance=0):
        """Check if return vector a dominates b."""
        # a dominates b if a is no worse than b in all objectives and better in at least one
        no_worse = jnp.all(a >= b - tolerance)
        better = jnp.any(a > b + tolerance)
        return no_worse and better
    
    def get_policy(self, weights=None):
        """Get policy from archive based on weights."""
        if len(self.policies) == 0:
            return None
            
        if weights is None:
            # Return random policy if no weights specified
            idx = np.random.randint(0, len(self.policies))
            return self.policies[idx][0]
            
        # Compute scalarized returns for all policies
        scalarized_returns = [jnp.dot(weights, r) for r in self.returns]
        
        # Find best policy according to scalarization
        best_idx = jnp.argmax(jnp.array(scalarized_returns))
        return self.policies[best_idx][0]
    
    def visualize_pareto_front(self):
        """Return the Pareto front for visualization."""
        return jnp.array(self.returns)
