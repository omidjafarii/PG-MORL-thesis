"""
Main script to run PG-MORL training with real-time visualization
Memory-optimized version
"""

import os
# Limit JAX memory usage to avoid consuming too much memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'

from morl_hopper import MorlHopper
from pgmorl_trainer import PGMORLTrainer
import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots without displaying them
import matplotlib.pyplot as plt
import numpy as np
import time
import jax
import gc
from matplotlib.animation import FuncAnimation

class LiveTrainingMonitor:
    def __init__(self):
        # Initialize lists to store metrics over time for visualization
        self.losses = []
        self.steps = []
        self.archive_sizes = []
        self.returns_history = []
        self.weights_history = []
        self.times = []
        
        # Create output directory for saving results
        os.makedirs('results', exist_ok=True)
        
        # Setup plot without interactive mode for efficient memory use
        self.fig, self.axs = plt.subplots(2, 1, figsize=(10, 12))
        self.fig.suptitle('PG-MORL Training Progress')
        
        # Plot for training loss
        self.loss_line, = self.axs[0].plot([], [], 'b-', label='Loss')
        self.axs[0].set_xlabel('Training Steps')
        self.axs[0].set_ylabel('Loss')
        self.axs[0].set_title('Training Loss')
        self.axs[0].grid(True)
        self.axs[0].legend()
        
        # Plot for Pareto archive size over training steps
        self.archive_line, = self.axs[1].plot([], [], 'r-', label='Archive Size')
        self.axs[1].set_xlabel('Training Steps')
        self.axs[1].set_ylabel('Archive Size')
        self.axs[1].set_title('Pareto Archive Growth')
        self.axs[1].grid(True)
        self.axs[1].legend()
        
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
    def update(self, step, loss, archive_size=None, returns=None, weights=None):
        """Update training monitor with new data."""
        # Append current metrics to the respective lists
        self.steps.append(step)
        self.losses.append(loss)
        self.times.append(time.time())
        
        if archive_size is not None:
            self.archive_sizes.append(archive_size)
            if returns is not None:
                self.returns_history.append(returns)
            if weights is not None:
                self.weights_history.append(weights)
        
        # Update the loss plot
        self.loss_line.set_data(self.steps, self.losses)
        self.axs[0].relim()
        self.axs[0].autoscale_view()
        
        # Update archive size plot periodically
        if self.archive_sizes:
            if len(self.steps) >= 20:
                archive_steps = [self.steps[i] for i in range(0, len(self.steps), 20)][:len(self.archive_sizes)]
            else:
                archive_steps = self.steps[:len(self.archive_sizes)]
                
            if len(archive_steps) != len(self.archive_sizes):
                archive_steps = self.steps[:len(self.archive_sizes)]
                
            self.archive_line.set_data(archive_steps, self.archive_sizes)
            self.axs[1].relim()
            self.axs[1].autoscale_view()
        
        # Save current plot (without showing it interactively)
        plt.savefig('results/training_progress.png')
        plt.close('all')  # Close plot to free memory
        
        # Recreate plot for the next update
        self.fig, self.axs = plt.subplots(2, 1, figsize=(10, 12))
        self.fig.suptitle('PG-MORL Training Progress')
        
        self.loss_line, = self.axs[0].plot(self.steps, self.losses, 'b-', label='Loss')
        self.axs[0].set_xlabel('Training Steps')
        self.axs[0].set_ylabel('Loss')
        self.axs[0].set_title('Training Loss')
        self.axs[0].grid(True)
        self.axs[0].legend()
        
        if self.archive_sizes:
            if len(self.steps) >= 20:
                archive_steps = [self.steps[i] for i in range(0, len(self.steps), 20)][:len(self.archive_sizes)]
            else:
                archive_steps = self.steps[:len(self.archive_sizes)]
            
            if len(archive_steps) != len(self.archive_sizes):
                archive_steps = self.steps[:len(self.archive_sizes)]

            self.archive_line, = self.axs[1].plot(archive_steps, self.archive_sizes, 'r-', label='Archive Size')
        else:
            self.archive_line, = self.axs[1].plot([], [], 'r-', label='Archive Size')
            
        self.axs[1].set_xlabel('Training Steps')
        self.axs[1].set_ylabel('Archive Size')
        self.axs[1].set_title('Pareto Archive Growth')
        self.axs[1].grid(True)
        self.axs[1].legend()
        
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Force garbage collection to manage memory
        gc.collect()
        
    def save_final_plots(self, archive):
        """Save all final plots after training completes."""
        # 2D scatter plots for each pair of objectives in the Pareto front
        obj_names = ['Forward Velocity', 'Control Cost', 'Alive Bonus']
        
        if len(archive) > 0:
            # Create Pareto front plots for each pair of objectives
            for i in range(3):
                for j in range(i+1, 3):
                    plt.figure(figsize=(8, 6))
                    returns = np.array(archive.returns)
                    plt.scatter(returns[:, i], returns[:, j], c='blue', s=50)
                    plt.xlabel(obj_names[i])
                    plt.ylabel(obj_names[j])
                    plt.title(f'Pareto Front: {obj_names[i]} vs {obj_names[j]}')
                    plt.grid(True)
                    plt.savefig(f'results/pareto_obj{i}_obj{j}.png')
                    plt.close()
                    gc.collect()  # Free memory after each plot
            
            # 3D plot of the Pareto front
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            returns = np.array(archive.returns)
            ax.scatter(returns[:, 0], returns[:, 1], returns[:, 2], c='red', s=50)
            
            ax.set_xlabel(obj_names[0])
            ax.set_ylabel(obj_names[1])
            ax.set_zlabel(obj_names[2])
            ax.set_title('3D Pareto Front')
            plt.savefig('results/pareto_3d.png')
            plt.close()
            gc.collect()


# Memory-optimized training function
def custom_train(trainer, monitor=None):
    """Main training loop with memory optimization."""
    print("Starting PG-MORL training...")
    
    # Initialize environment
    key, reset_key = jax.random.split(trainer.key)
    reset_key = jax.random.split(reset_key, trainer.num_envs)
    env_state = trainer.env.reset(reset_key)
    
    # Training metrics
    metrics = {'step': 0, 'weights': []}
    start_time = time.time()
    
    # Training loop for the given number of timesteps
    num_updates = trainer.num_timesteps // (trainer.num_envs * trainer.unroll_length)
    print(f"Total updates to perform: {num_updates}")
    
    # Add progress counter
    last_print_time = time.time()
    
    for update in range(num_updates):
        print(f"Starting update {update+1}, collecting batch...")
        
        # Sample new weights periodically to change objective emphasis
        if update % 10 == 0:
            key, weights_key = jax.random.split(key)
            weights = np.random.dirichlet(np.ones(trainer.num_objectives))
            # Update environment weights for scalarization
            trainer.env.weights = jax.numpy.array(weights)
            metrics['weights'].append(weights)
        
        # Collect batch of trajectories
        key, env_state, batch = trainer._sample_batch(key, trainer.network_params, env_state)
        
        print(f"Batch collected, starting network updates...")
        
        # Prepare batch for updates
        batch_size = trainer.num_envs * trainer.unroll_length
        batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
            
        # Normalize advantages for stable learning
        batch['advantage'] = (batch['advantage'] - batch['advantage'].mean()) / (batch['advantage'].std() + 1e-8)
        
        # Update policy and value function using mini-batches
        for _ in range(trainer.num_updates_per_batch):
            key, subkey = jax.random.split(key)
            permutation = jax.random.permutation(subkey, batch_size)
            
            # Mini-batch updates for policy and value function
            for start in range(0, batch_size, trainer.batch_size):
                end = min(start + trainer.batch_size, batch_size)
                minibatch_indices = permutation[start:end]
                
                minibatch = jax.tree_util.tree_map(lambda x: x[minibatch_indices], batch)
                
                trainer.network_params, trainer.optimizer_state, update_metrics = trainer._update_step(
                    trainer.network_params,
                    trainer.optimizer_state,
                    minibatch['obs'],
                    minibatch['action'],
                    minibatch['advantage'],
                    minibatch['return'],
                    minibatch['log_prob']
                )
                
                metrics.update(update_metrics)
        
        # Free up memory
        batch = None
        minibatch = None
        gc.collect()
        
        print(f"Network updated, starting evaluation (if scheduled)...")
        
        # Print progress more frequently
        current_time = time.time()
        if update % 2 == 0 or (current_time - last_print_time) > 10:
            elapsed_time = current_time - start_time
            steps = (update + 1) * trainer.num_envs * trainer.unroll_length
            metrics['step'] = steps
            
            print(f"Update: {update+1}/{num_updates} ({(update+1)/num_updates*100:.1f}%) - "
                  f"Steps: {steps}/{trainer.num_timesteps} - "
                  f"Time: {elapsed_time:.2f}s - "
                  f"Loss: {metrics['loss']:.4f}")
            
            # Update the monitor if provided
            if monitor:
                monitor.update(steps, metrics['loss'])
                
            last_print_time = current_time
        
        # Less frequent evaluation to save memory
        if update % 20 == 0:
            print("Running evaluation...")
            returns = trainer._evaluate_policy(trainer.network_params, num_episodes=5)  # Reduced episodes
            trainer.archive.update(trainer.network_params, returns)
            
            # Log detailed metrics after evaluation
            print(f"  > Archive Size: {len(trainer.archive)}")
            print(f"  > Current weights: {weights}")
            print(f"  > Average returns: {returns.mean(axis=0)}")
            
            # Update monitor with archive info
            if monitor:
                monitor.update(steps, metrics['loss'], len(trainer.archive), 
                              returns.mean(axis=0), weights)
            
            # Free memory after evaluation
            gc.collect()
    
    print("Training complete!")
    
    # Return trained params and Pareto archive
    return trainer.network_params, trainer.archive


def main():
    # Create the live training monitor for visualization
    monitor = LiveTrainingMonitor()
    
    # Create the trainer with memory-optimized settings
    trainer = PGMORLTrainer(
        env_fn=MorlHopper,
        num_timesteps=5000,       # Reduced timesteps for memory optimization
        episode_length=200,       # Shorter episodes to reduce computation
        action_repeat=1,
        num_envs=1,               # Minimal parallelism for memory efficiency
        num_eval_envs=2,          # Fewer eval environments
        learning_rate=3e-4,
        entropy_cost=1e-2,
        unroll_length=10,         # Shorter unroll length to manage memory
        batch_size=16,            # Smaller batch size to fit in memory
        num_objectives=3
    )
    
    # Start training and monitoring
    try:
        trained_params, archive = custom_train(trainer, monitor)
        
        # Visualize and log Pareto front after training
        pareto_front = archive.visualize_pareto_front()
        print(f"Pareto front contains {len(archive)} policies")
        
        # Save archive to file
        with open('pgmorl_archive.pkl', 'wb') as f:
            pickle.dump({'params': archive.policies, 'returns': archive.returns}, f)
        
        # Create final Pareto front visualizations
        monitor.save_final_plots(archive)
        
        print("Training and visualization complete!")
        
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving current results...")
        # Save partial results if interrupted
        if hasattr(trainer, 'archive'):
            with open('pgmorl_archive_partial.pkl', 'wb') as f:
                pickle.dump({'params': trainer.archive.policies, 
                           'returns': trainer.archive.returns}, f)
            monitor.save_final_plots(trainer.archive)
            print(f"Partial results saved. Archive size: {len(trainer.archive)}")

if __name__ == "__main__":
    main()
