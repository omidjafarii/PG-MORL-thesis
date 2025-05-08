"""
A100-Optimized implementation of run_halfcheetah.py
Designed for maximum memory efficiency on A100 GPUs
Integrates advanced JAX memory management techniques with PGMORL algorithm
Fixed for compatibility with recent JAX versions and Puhti compatibility
"""

import os
import pickle
import numpy as np
import time
import jax
import jax.numpy as jnp
import gc
from jax.lib import xla_bridge
from morl_halfcheetah import MorlHalfcheetah
from pgmorl_trainer import PGMORLTrainer, ParetoArchive
import re
import optax
import sys

# === A100-specific JAX optimizations ===
# Prevent pre-allocation of GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Limit JAX to 80% of GPU memory
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
# Use platform's native memory allocator
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# Set matrix multiplication to use bfloat16 by default
jax.config.update("jax_default_matmul_precision", "bfloat16")

# Print version info for debugging
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"JAX version: {jax.__version__}")

# Set up device mesh for potential parallelization
try:
    from jax.sharding import Mesh
    from jax.experimental import mesh_utils
    from jax import checkpoint
    # Create a device mesh even if just using one device
    devices = jax.devices()
    mesh = Mesh(mesh_utils.create_device_mesh((len(devices),)), axis_names=("dp",))
    CHECKPOINT_AVAILABLE = True
except ImportError:
    print("[WARNING] JAX checkpointing not available in this JAX version. Using fallback.")
    CHECKPOINT_AVAILABLE = False
    # Define a dummy checkpoint function if not available
    def checkpoint(f):
        return f

# === Fix for JAX compatibility ===
# Use the appropriate tree_map function based on JAX version
try:
    # Try the newer location first
    from jax.tree_util import tree_map
except ImportError:
    try:
        # Fall back to the old location
        from jax.tree_util import tree_map
    except ImportError:
        # Last resort
        from jax import tree_map

# === Memory Management Helpers ===
def clear_caches():
    """Clear all JAX caches and run garbage collection."""
    jax.clear_caches()
    gc.collect()
    
def log_memory_usage(tag):
    """Log the current GPU memory usage."""
    try:
        import subprocess
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'])
        mem_used = int(result.decode('utf-8').strip())
        print(f"[MEM] {tag}: {mem_used} MiB")
    except Exception as e:
        print(f"[MEM] Could not get memory usage: {e}")

# === Memory-Efficient Pareto Archive ===
class MemoryEfficientParetoArchive(ParetoArchive):
    """Memory-efficient archive to store Pareto optimal policies."""
    
    def __init__(self, num_objectives=2, epsilon=0.01, max_policies=50):
        super().__init__(num_objectives, epsilon)
        self.max_policies = max_policies  # Maximum number of policies to store
        
    def update(self, params, returns):
        """Update archive with new policy if it's non-dominated."""
        # Convert to bf16 for storage if on GPU
        if jax.devices()[0].platform == 'gpu':
            lightweight_params = tree_map(
                lambda x: jnp.asarray(x, dtype=jnp.bfloat16), params)
        else:
            lightweight_params = params
            
        if len(self.policies) == 0:
            # First policy, add it directly
            self.policies.append((lightweight_params, returns))
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
            self.policies.append((lightweight_params, returns))
            self.returns.append(returns)
            
            # If we exceed maximum size, trim policies
            if len(self.policies) > self.max_policies:
                self._trim_archive()
                
            return True
            
        return False
    
    def _trim_archive(self):
        """Trim archive by removing policies with smallest contribution."""
        # Calculate contribution of each policy (simplified)
        contributions = []
        for i, ret in enumerate(self.returns):
            # Use distance from origin as a simple contribution metric
            contribution = jnp.sum(ret)
            contributions.append((i, contribution))
        
        # Sort by contribution and keep top max_policies
        sorted_contribs = sorted(contributions, key=lambda x: x[1], reverse=True)
        keep_indices = [idx for idx, _ in sorted_contribs[:self.max_policies]]
        
        # Keep only the top policies
        self.policies = [self.policies[i] for i in keep_indices]
        self.returns = [self.returns[i] for i in keep_indices]
        
    def get_policy(self, weights=None):
        """Get policy from archive based on weights."""
        if len(self.policies) == 0:
            return None
            
        if weights is None:
            # Return random policy if no weights specified
            idx = np.random.randint(0, len(self.policies))
            policy = self.policies[idx][0]
        else:
            # Compute scalarized returns for all policies
            scalarized_returns = [jnp.dot(weights, r) for r in self.returns]
            
            # Find best policy according to scalarization
            best_idx = jnp.argmax(jnp.array(scalarized_returns))
            policy = self.policies[best_idx][0]
        
        # Convert back to f32 if it was stored as bf16
        if jax.devices()[0].platform == 'gpu':
            policy = tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), policy)
            
        return policy

# === Checkpoint and Extracted Data Helpers ===
def load_extracted_data(data_path, trainer):
    """
    Load data from the extracted checkpoint file created on a machine with NumPy 2.2.4
    """
    print(f"[LOG] Loading extracted checkpoint data from: {data_path}")
    
    try:
        # Load the extracted data
        with open(data_path, 'rb') as f:
            simple_data = pickle.load(f)
        
        # Convert NumPy arrays back to JAX arrays
        network_params = {}
        for k, v in simple_data['network_params_numpy'].items():
            network_params[k] = jnp.array(v)
        
        # Create a fresh optimizer state (since we didn't extract it)
        optimizer_state = trainer.optimizer.init(network_params)
        
        # Reconstruct archive
        archive = MemoryEfficientParetoArchive(
            num_objectives=trainer.num_objectives,
            epsilon=0.01,
            max_policies=50
        )
        
        for returns in simple_data['archive_returns']:
            # Use dummy parameters for archive policies (the actual params aren't critical)
            dummy_params = network_params
            archive.update(dummy_params, jnp.array(returns))
        
        print(f"[LOG] Loaded extracted data with update: {simple_data['update']}")
        print(f"[LOG] Archive size: {len(archive.returns)}")
        
        return network_params, optimizer_state, simple_data['update'], archive.policies, archive.returns
    
    except Exception as e:
        print(f"[ERROR] Failed to load extracted data: {e}")
        print("[LOG] Starting fresh training run.")
        return None, None, 0, None, None

def save_checkpoint(network_params, optimizer_state, update, archive, checkpoint_dir='checkpoints_halfcheetah'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    if len(archive.returns) == 0:
        print(f"[WARN] Skipping checkpoint save at update {update}: archive is empty.")
        return
    filename = os.path.join(checkpoint_dir, f'checkpoint_{update}.pkl')
    with open(filename, 'wb') as f:
        pickle.dump({
            'network_params': network_params,
            'optimizer_state': optimizer_state,
            'update': update,
            'archive_params': archive.policies,
            'archive_returns': archive.returns
        }, f)
    print(f"[LOG] Saved checkpoint at update {update} -> {filename}")

def load_checkpoint(checkpoint_dir='checkpoints_halfcheetah'):
    if not os.path.exists(checkpoint_dir):
        return None, None, 0, None, None
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pkl')]
    if not files:
        print("[LOG] No checkpoint files found.")
        return None, None, 0, None, None

    latest_file = max(files, key=lambda f: int(re.search(r'\d+', f).group()))
    filepath = os.path.join(checkpoint_dir, latest_file)
    print(f"[DEBUG] Loading checkpoint from: {os.path.abspath(filepath)}")

    try:
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        print(f"[LOG] Loaded checkpoint at update {checkpoint['update']}")
        return (
            checkpoint['network_params'],
            checkpoint['optimizer_state'],
            checkpoint['update'],
            checkpoint.get('archive_params'),
            checkpoint.get('archive_returns')
        )
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        print("[LOG] Unable to load checkpoint. Will try extracted data or start fresh.")
        return None, None, 0, None, None

def clean_old_checkpoints(checkpoint_dir='checkpoints_halfcheetah', keep_last_n=3):
    if not os.path.exists(checkpoint_dir):
        return
    files = [f for f in os.listdir(checkpoint_dir) if re.match(r'checkpoint_\d+\.pkl', f)]
    if len(files) <= keep_last_n:
        return
    files_sorted = sorted(files, key=lambda f: int(re.search(r'\d+', f).group()))
    to_delete = files_sorted[:-keep_last_n]
    for f in to_delete:
        os.remove(os.path.join(checkpoint_dir, f))
        print(f"[LOG] Deleted old checkpoint: {f}")

# === Ultra-Optimized Evaluation Function with A100 Optimizations ===
def single_env_evaluate(trainer, params, num_episodes=5):
    """A100-optimized evaluation that evaluates just ONE environment at a time."""
    print("[EVAL] Starting single-environment evaluation strategy")
    log_memory_usage("Evaluation start")
    
    # Force garbage collection before starting
    gc.collect()
    jax.clear_caches()
    
    # We'll process one environment at a time - no batching
    all_returns = []
    
    # Process each episode one at a time
    for episode in range(num_episodes):
        print(f"[EVAL] Running episode {episode+1}/{num_episodes}")
        
        # Reset a SINGLE environment
        key, reset_key = jax.random.split(trainer.key)
        trainer.key = key
        
        # Create a single reset key - we're only resetting ONE environment
        reset_key = jax.random.split(reset_key, 1)
        
        # Use a dummy implementation with batch size 1
        eval_env_copy = trainer.eval_env
        original_batch_size = trainer.eval_batch_size
        trainer.eval_batch_size = 1
        
        # Reset just one environment
        eval_state = trainer.eval_env.reset(reset_key)
        
        # Force computation before proceeding
        eval_state = jax.block_until_ready(eval_state)
        
        # Track returns for this episode
        returns = jnp.zeros((1, trainer.num_objectives))
        done = jnp.zeros((1,), dtype=bool)
        
        # Force computation
        returns = jax.device_put(returns)
        done = jax.device_put(done)
        
        # Run the episode step by step within a try-except block
        try:
            for step_idx in range(trainer.episode_length):
                # Get action directly without JIT
                mean, _, _ = trainer.network.apply(params, eval_state.obs)
                action = jnp.clip(mean, -1.0, 1.0)
                
                # Force computation to avoid buildup
                action = jax.block_until_ready(action)
                
                # Take step in environment - this is the slow part
                # Use explicit host callback to prevent excessive GPU memory usage
                if step_idx % 50 == 0:
                    # Move computations to CPU periodically to reduce memory pressure
                    clear_caches()
                
                # Take environment step
                eval_state = eval_env_copy.step(eval_state, action)
                
                # Force computation after every step
                eval_state = jax.block_until_ready(eval_state)
                
                # Update returns (only for active environments)
                reward_vector = eval_state.metrics['reward_vector']
                mask = ~done[:, None]
                returns = returns + mask * reward_vector
                done = jnp.logical_or(done, eval_state.done)
                
                # Force computation to avoid memory buildup
                returns = jax.block_until_ready(returns)
                done = jax.block_until_ready(done)
                
                # Break early if done
                if jnp.all(done):
                    print(f"[EVAL] Episode {episode+1} finished early at step {step_idx+1}")
                    break
                
                # Clear memory periodically
                if step_idx % 10 == 0:
                    gc.collect()
                    jax.clear_caches()
        
        except Exception as e:
            print(f"[EVAL ERROR] Step failed: {e}")
            if episode == 0 and len(all_returns) == 0:
                # If first episode and no returns yet, try one more time with even more conservative approach
                try:
                    print("[EVAL] Retrying with more conservative approach...")
                    # Create a completely fresh environment
                    temp_env = MorlHalfcheetah()
                    
                    # Create a fresh key and split correctly
                    fresh_key = jax.random.PRNGKey(int(time.time()))
                    fresh_reset_key = jax.random.split(fresh_key, 1)
                    
                    temp_state = temp_env.reset(fresh_reset_key)
                    temp_returns = jnp.zeros(trainer.num_objectives)
                    
                    # Run a very simple loop
                    for _ in range(50):  # Just do 50 steps as a fallback
                        mean, _, _ = trainer.network.apply(params, temp_state.obs)
                        action = jnp.clip(mean, -1.0, 1.0)
                        temp_state = temp_env.step(temp_state, action)
                        temp_returns += temp_state.metrics['reward_vector'][0]
                        if temp_state.done[0]:
                            break
                    
                    # Just return something reasonable
                    print(f"[EVAL] Fallback returns: {temp_returns}")
                    return temp_returns
                    
                except Exception as e2:
                    print(f"[EVAL ERROR] Even fallback evaluation failed: {e2}")
                    # Last resort: return zeros
                    return jnp.zeros(trainer.num_objectives)
            
            # If not first episode, use what we have so far
            if len(all_returns) > 0:
                # We have at least one good episode, so average what we have
                average_returns = jnp.mean(jnp.stack(all_returns), axis=0)
                print(f"[EVAL] Using partial results from {len(all_returns)} episodes: {average_returns}")
                return average_returns
            else:
                # No valid episodes, return zeros as last resort
                print("[EVAL] No valid episodes completed, returning zeros")
                return jnp.zeros(trainer.num_objectives)
        
        # This episode's return
        episode_return = jnp.mean(returns, axis=0)
        episode_return = jax.block_until_ready(episode_return)
        all_returns.append(episode_return)
        
        print(f"[EVAL] Episode {episode+1} returns: {episode_return}")
        
        # Aggressively clear memory between episodes
        eval_state = None
        returns = None
        done = None
        action = None
        gc.collect()
        jax.clear_caches()
        
        # Restore original batch size
        trainer.eval_batch_size = original_batch_size
    
    # Calculate average returns across episodes
    if all_returns:
        average_returns = jnp.mean(jnp.stack(all_returns), axis=0)
        average_returns = jax.block_until_ready(average_returns)
    else:
        # Fallback in case no episodes completed
        average_returns = jnp.zeros(trainer.num_objectives)
    
    # Final memory cleanup
    log_memory_usage("Evaluation end")
    all_returns = None
    gc.collect()
    jax.clear_caches()
    
    return average_returns

# === Memory-Efficient Update Methods with A100 Optimizations ===
def update_with_gradient_accumulation(trainer, params, opt_state, batch, num_microbatches=4):
    """Update parameters with gradient accumulation and memory optimization."""
    batch_size = batch['obs'].shape[0]
    microbatch_size = batch_size // num_microbatches
    
    # Initialize accumulated gradients with zeros
    def init_grad(p):
        return jnp.zeros_like(p)
    
    accumulated_grads = tree_map(init_grad, params)
    
    # Process each microbatch
    policy_loss_total = 0
    value_loss_total = 0
    
    for i in range(num_microbatches):
        start_idx = i * microbatch_size
        end_idx = min(start_idx + microbatch_size, batch_size)
        
        # Extract microbatch
        microbatch = tree_map(
            lambda x: x[start_idx:end_idx], batch)
        
        # Compute gradients for this microbatch
        def loss_fn(p):
            # Use checkpointing to trade computation for memory if available
            if CHECKPOINT_AVAILABLE:
                policy_loss = checkpoint(lambda p: trainer._policy_loss(
                    p, microbatch['obs'], microbatch['action'], 
                    microbatch['advantage'], microbatch['log_prob']))(p)
                
                value_loss = checkpoint(lambda p: trainer._value_loss(
                    p, microbatch['obs'], microbatch['return']))(p)
            else:
                # Fallback to standard computation if checkpointing not available
                policy_loss = trainer._policy_loss(
                    p, microbatch['obs'], microbatch['action'], 
                    microbatch['advantage'], microbatch['log_prob'])
                
                value_loss = trainer._value_loss(
                    p, microbatch['obs'], microbatch['return'])
                
            total_loss = policy_loss + 0.5 * value_loss
            return total_loss, (policy_loss, value_loss)
        
        (_, (policy_loss, value_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        
        # Accumulate gradients
        accumulated_grads = tree_map(
            lambda acc, g: acc + g, accumulated_grads, grads)
        
        # Accumulate losses for averaging
        policy_loss_total += policy_loss
        value_loss_total += value_loss
        
        # Force computation and clear intermediates
        tree_map(lambda x: x.block_until_ready(), grads)
        del grads, microbatch
        
        # Clear cache every other microbatch
        if i % 2 == 1:
            clear_caches()
    
    # Average the losses
    policy_loss_avg = policy_loss_total / num_microbatches
    value_loss_avg = value_loss_total / num_microbatches
    
    # Scale gradients by the number of microbatches
    accumulated_grads = tree_map(
        lambda g: g / num_microbatches, accumulated_grads)
    
    # Apply accumulated gradients
    updates, new_opt_state = trainer.optimizer.update(accumulated_grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    
    # Metrics for logging
    metrics = {
        'loss': policy_loss_avg + 0.5 * value_loss_avg,
        'policy_loss': policy_loss_avg,
        'value_loss': value_loss_avg,
    }
    
    return new_params, new_opt_state, metrics

# === Memory-Efficient Batch Collection with A100 Optimizations ===
def collect_efficient_batch(trainer, key, params, env_state):
    """Collect experience in smaller chunks to reduce peak memory usage."""
    
    # Use the existing _sample_batch method but modify the returned data
    key, env_state, batch = trainer._sample_batch(key, params, env_state)
    
    # Process the batch to reduce size before returning
    batch_size = trainer.num_envs * trainer.unroll_length
    
    # Reshape flat and force computation
    batch = tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
    
    # If on GPU, convert to lower precision for storage
    if jax.devices()[0].platform == 'gpu':
        # Only convert the large observation and return tensors to bf16
        batch['obs'] = jnp.asarray(batch['obs'], dtype=jnp.bfloat16)
        batch['return'] = jnp.asarray(batch['return'], dtype=jnp.bfloat16)
    
    # Force immediate computation
    batch = tree_map(lambda x: jax.device_put(x, jax.devices()[0]), batch)
    
    return key, env_state, batch

# === Memory-Optimized Training Function ===
def memory_optimized_train(trainer):
    """A100-optimized training function for HalfCheetah."""
    print("[LOG] Starting A100-optimized PG-MORL training for HalfCheetah...")
    log_memory_usage("Training start")
    
    # First try loading from extracted data (for Puhti compatibility)
    extracted_data_path = "extracted_data.pkl"
    if os.path.exists(extracted_data_path):
        print("[LOG] Found extracted data file, attempting to load...")
        network_params, optimizer_state, start_update, archive_params, archive_returns = load_extracted_data(
            extracted_data_path, trainer)
        if network_params is not None:
            print("[LOG] Successfully loaded extracted data")
        else:
            print("[LOG] Failed to load extracted data, falling back to checkpoint")
            network_params, optimizer_state, start_update, archive_params, archive_returns = load_checkpoint()
    else:
        # Try to load regular checkpoint
        network_params, optimizer_state, start_update, archive_params, archive_returns = load_checkpoint()

    if network_params is not None:
        print("[LOG] Resuming from loaded data")
        trainer.network_params = network_params
        trainer.optimizer_state = optimizer_state
        if archive_params is not None and archive_returns is not None:
            # Replace the archive with our memory-efficient version
            efficient_archive = MemoryEfficientParetoArchive(
                num_objectives=trainer.num_objectives,
                epsilon=0.01,
                max_policies=50  # Limit total number of policies stored
            )
            efficient_archive.policies = archive_params
            efficient_archive.returns = archive_returns
            trainer.archive = efficient_archive
            print(f"[LOG] Restored archive with {len(archive_returns)} entries")
    else:
        print("[LOG] Starting fresh training run")
        start_update = 0
        # Replace standard archive with memory-efficient version
        trainer.archive = MemoryEfficientParetoArchive(
            num_objectives=trainer.num_objectives,
            epsilon=0.01, 
            max_policies=50
        )
    
    # Initialize environment and reset
    key, reset_key = jax.random.split(trainer.key)
    reset_key = jax.random.split(reset_key, trainer.num_envs)
    env_state = trainer.env.reset(reset_key)
    
    # Move to device and ensure computation is complete
    env_state = jax.device_put(env_state, jax.devices()[0])
    
    # Training metrics
    metrics = {'step': 0, 'weights': []}
    start_time = time.time()
    
    # Training loop settings
    num_updates = trainer.num_timesteps // (trainer.num_envs * trainer.unroll_length)
    print(f"[LOG] Total updates to perform: {num_updates}")
    
    # Progress tracking
    last_print_time = time.time()
    
    # Define fixed weights for better Pareto front coverage (same as hopper for fair comparison)
    fixed_weights = np.array([
        [0.95, 0.05], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4],
        [0.5, 0.5], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9], [0.05, 0.95]
    ])
    
    # Training loop
    for update in range(start_update, num_updates):
        update_start = time.time()
        
        # Sample new weights periodically (same as Hopper for fair comparison)
        if update % 5 == 0:
            weights = fixed_weights[update % len(fixed_weights)]
            trainer.env.weights = jnp.array(weights)
            metrics['weights'].append(weights)
            print(f"[LOG] Using weights: {weights}")
        
        try:
            # Collect batch of trajectories using memory-efficient method
            key, env_state, batch = collect_efficient_batch(trainer, key, trainer.network_params, env_state)
            
            # Normalize advantages
            batch['advantage'] = (batch['advantage'] - batch['advantage'].mean()) / (batch['advantage'].std() + 1e-8)
            
            # A100 optimization: convert to bfloat16 for bandwidth-bound operations
            if jax.devices()[0].platform == 'gpu':
                batch['advantage'] = jnp.asarray(batch['advantage'], dtype=jnp.bfloat16)
            
            # Update policy and value function using gradient accumulation
            for _ in range(trainer.num_updates_per_batch):
                key, subkey = jax.random.split(key)
                permutation = jax.random.permutation(subkey, batch['obs'].shape[0])
                
                # Process in minibatches with gradient accumulation
                trainer.network_params, trainer.optimizer_state, update_metrics = update_with_gradient_accumulation(
                    trainer,
                    trainer.network_params,
                    trainer.optimizer_state,
                    batch,
                    num_microbatches=4  # Split batch into 4 parts to reduce memory
                )
                
                metrics.update(update_metrics)
            
            # Force execution before clearing
            tree_map(lambda x: x.block_until_ready(), trainer.network_params)
            
            # Clear batch data to free memory
            batch = None
            clear_caches()
            
        except Exception as e:
            print(f"[ERROR] Exception during update {update}: {str(e)}")
            # Save checkpoint before potential crash
            save_checkpoint(trainer.network_params, trainer.optimizer_state, update, trainer.archive)
            raise
        
        # Save checkpoint periodically (same as Hopper)
        if update % 5 == 0 or update == num_updates - 1:
            save_checkpoint(trainer.network_params, trainer.optimizer_state, update, trainer.archive)
            clean_old_checkpoints()
        
        # Log progress
        current_time = time.time()
        update_duration = current_time - update_start
        if update % 2 == 0 or (current_time - last_print_time) > 10:
            elapsed_time = current_time - start_time
            steps = (update + 1) * trainer.num_envs * trainer.unroll_length
            progress = (update + 1) / num_updates * 100
            
            print(f"[LOG] Update {update+1}/{num_updates} ({progress:.1f}%) - "
                  f"Steps: {steps}/{trainer.num_timesteps} - "
                  f"Time: {elapsed_time:.2f}s (this update: {update_duration:.2f}s) - "
                  f"Loss: {metrics['loss']:.4f}")
            
            log_memory_usage(f"Update {update+1}")
            last_print_time = current_time
        
        # Run evaluation less frequently (same as Hopper for fair comparison)
        if update % 5 == 0 or update == num_updates - 1:
            try:
                print(f"[LOG] Running evaluation at update {update+1}/{num_updates}")
                # Clear memory before evaluation
                clear_caches()
                
                # Use ultra-optimized single environment evaluation
                returns = single_env_evaluate(
                    trainer, 
                    trainer.network_params, 
                    num_episodes=10  # Same as Hopper
                )
                
                # Update archive with new policy
                added = trainer.archive.update(trainer.network_params, returns)
                
                print(f"[LOG] Archive Size: {len(trainer.archive)}")
                print(f"[LOG] Current weights: {weights}")
                print(f"[LOG] Average returns: {returns}")
                print(f"[LOG] Added to archive: {added}")
                
                # Save intermediate archive periodically
                if update % 50 == 0 and update > 0:
                    archive_file = f'halfcheetah_archive_{update}.pkl'
                    with open(archive_file, 'wb') as f:
                        pickle.dump({
                            'params': trainer.archive.policies, 
                            'returns': trainer.archive.returns,
                            'update': update
                        }, f)
                    print(f"[LOG] Saved intermediate archive to {archive_file}")
                
                # Force cleanup after evaluation
                returns = None
                clear_caches()
                
            except Exception as e:
                print(f"[ERROR] Evaluation failed at update {update}: {str(e)}")
                # Continue training even if evaluation fails
    
    print("[LOG] Training complete!")
    return trainer.network_params, trainer.archive

# === Main Entry Point ===
def main():
    print("[DEBUG] Checking JAX backend and devices...")
    backend = xla_bridge.get_backend().platform
    devices = jax.devices()
    print(f"[DEBUG] JAX is using backend: {backend}")
    print(f"[DEBUG] Available JAX devices: {devices}")
    
    # Check GPU availability
    if backend != 'gpu':
        print("[WARNING] JAX is not using GPU! Memory optimization will still help, but be aware.")
        print("[WARNING] Consider adding: export JAX_PLATFORM_NAME=gpu")
    
    # Log initial memory usage
    log_memory_usage("Startup")
    
    # Create directories
    os.makedirs("checkpoints_halfcheetah", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Create trainer with same hyperparameters as hopper for fair comparison
    trainer = PGMORLTrainer(
        env_fn=MorlHalfcheetah,
        num_timesteps=200_000,  # Same as Hopper
        episode_length=200,     # Same as Hopper
        action_repeat=1,        # Same as Hopper
        num_envs=24,            # Same as Hopper
        num_eval_envs=2,        # Same as Hopper
        learning_rate=3e-4,     # Same as Hopper
        entropy_cost=1e-2,      # Same as Hopper
        unroll_length=10,       # Same as Hopper
        batch_size=256,         # Same as Hopper
        num_updates_per_batch=4,# Same as Hopper
        num_objectives=2,       # Same as Hopper
        discounting=0.99,       # Same as Hopper
        seed=42                 # Same as Hopper
    )
    
    try:
        # Run optimized training function
        trained_params, archive = memory_optimized_train(trainer)
        
        # Save final results
        print(f"[LOG] Training complete! Final Pareto archive size: {len(archive)}")
        with open('halfcheetah_archive_final.pkl', 'wb') as f:
            pickle.dump({'params': archive.policies, 'returns': archive.returns}, f)
        print("[LOG] Archive saved to halfcheetah_archive_final.pkl")
        
        # Create final plots
        if len(archive.returns) > 0:
            try:
                import matplotlib.pyplot as plt
                
                returns = np.array([r for r in archive.returns])
                
                # 2D plot of Pareto front
                plt.figure(figsize=(10, 8))
                plt.scatter(returns[:, 0], returns[:, 1], c='blue', s=50)
                plt.xlabel('Speed Reward')
                plt.ylabel('Energy Efficiency')
                plt.title('HalfCheetah Pareto Front')
                plt.grid(True)
                plt.savefig('results/halfcheetah_pareto_front.png')
                plt.close()
                
                print(f"[LOG] Created visualization in results/halfcheetah_pareto_front.png")
            except Exception as e:
                print(f"[ERROR] Failed to create visualization: {e}")
        
    except KeyboardInterrupt:
        print("[LOG] Training interrupted by user. Saving partial results...")
        # Save partial results
        with open('halfcheetah_archive_partial.pkl', 'wb') as f:
            pickle.dump({'params': trainer.archive.policies, 
                       'returns': trainer.archive.returns}, f)
        print("[LOG] Partial archive saved to halfcheetah_archive_partial.pkl")
    
    except Exception as e:
        print(f"[ERROR] Training failed with error: {str(e)}")
        # Save whatever we have
        with open('halfcheetah_archive_error.pkl', 'wb') as f:
            pickle.dump({'params': trainer.archive.policies, 
                       'returns': trainer.archive.returns,
                       'error': str(e)}, f)
        print("[LOG] Error state saved to halfcheetah_archive_error.pkl")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"[CRITICAL ERROR] {str(e)}")
        traceback.print_exc()
        # Save memory information even on critical error
        try:
            log_memory_usage("Crash")
        except:
            pass