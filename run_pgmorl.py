import os
import pickle
import numpy as np
import time
import jax
import gc
from jax.lib import xla_bridge
from morl_hopper import MorlHopper
from pgmorl_trainer import PGMORLTrainer
import re

# === Checkpoint Helpers ===
def save_checkpoint(network_params, optimizer_state, update, archive, checkpoint_dir='checkpoints'):
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
    print(f"[LOG] ? Saved checkpoint at update {update} -> {filename}")

def load_checkpoint(checkpoint_dir='checkpoints'):
    if not os.path.exists(checkpoint_dir):
        return None, None, 0, None, None
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pkl')]
    if not files:
        print("[LOG] ?? No checkpoint files found.")
        return None, None, 0, None, None

    latest_file = max(files, key=lambda f: int(f.split('_')[1].split('.')[0]))
    filepath = os.path.join(checkpoint_dir, latest_file)
    print(f"[DEBUG] Loading checkpoint from: {os.path.abspath(filepath)}")

    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)

    print(f"[LOG] ?? Loaded checkpoint at update {checkpoint['update']}")
    return (
        checkpoint['network_params'],
        checkpoint['optimizer_state'],
        checkpoint['update'],
        checkpoint.get('archive_params'),
        checkpoint.get('archive_returns')
    )

def clean_old_checkpoints(checkpoint_dir='checkpoints', keep_last_n=3):
    if not os.path.exists(checkpoint_dir):
        return
    files = [f for f in os.listdir(checkpoint_dir) if re.match(r'checkpoint_\d+\.pkl', f)]
    if len(files) <= keep_last_n:
        return
    files_sorted = sorted(files, key=lambda f: int(re.search(r'\d+', f).group()))
    to_delete = files_sorted[:-keep_last_n]
    for f in to_delete:
        os.remove(os.path.join(checkpoint_dir, f))
        print(f"[LOG] ?? Deleted old checkpoint: {f}")

# === Training ===
def custom_train(trainer):
    print("[LOG] Starting PG-MORL training (final run)...")

    network_params, optimizer_state, start_update, archive_params, archive_returns = load_checkpoint()

    if network_params is not None:
        trainer.network_params = network_params
        trainer.optimizer_state = optimizer_state
        if archive_params is not None and archive_returns is not None:
            trainer.archive.policies = archive_params
            trainer.archive.returns = archive_returns
            print(f"[LOG] ✅ Restored archive with {len(archive_returns)} entries")
    else:
        start_update = 0

    key, reset_key = jax.random.split(trainer.key)
    reset_key = jax.random.split(reset_key, trainer.num_envs)
    env_state = trainer.env.reset(reset_key)
    env_state = jax.device_put(env_state, jax.devices()[0])  # ✅ Move env to GPU

    metrics = {'step': 0, 'weights': []}
    start_time = time.time()
    num_updates = trainer.num_timesteps // (trainer.num_envs * trainer.unroll_length)
    print(f"[LOG] Total updates to perform: {num_updates}")
    last_print_time = time.time()

    fixed_weights = np.array([
        [0.95, 0.05], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4],
        [0.5, 0.5], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9], [0.05, 0.95]
    ])

    for update in range(start_update, num_updates):
        if update % 5 == 0:
            weights = fixed_weights[update % len(fixed_weights)]
            trainer.env.weights = jax.numpy.array(weights)
            metrics['weights'].append(weights)
            print(f"[LOG] Sampled weights: {weights}")

        key, env_state, batch = trainer._sample_batch(key, trainer.network_params, env_state)
        batch_size = trainer.num_envs * trainer.unroll_length
        batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)

        # ✅ Move all batch data to GPU
        batch = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices()[0]), batch)

        # ✅ Check one example to confirm device
        try:
            print("[DEBUG] batch['obs'] is on:", batch['obs'].device_buffer.device())
        except:
            print("[DEBUG] Could not print device info.")

        # Normalize advantage
        batch['advantage'] = (batch['advantage'] - batch['advantage'].mean()) / (batch['advantage'].std() + 1e-8)

        for _ in range(trainer.num_updates_per_batch):
            key, subkey = jax.random.split(key)
            permutation = jax.random.permutation(subkey, batch_size)
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

        # ✅ Save every 5 updates
        if update % 5 == 0:
            save_checkpoint(trainer.network_params, trainer.optimizer_state, update, trainer.archive)
            clean_old_checkpoints(keep_last_n=3)

        gc.collect()

        current_time = time.time()
        if update % 2 == 0 or (current_time - last_print_time) > 10:
            elapsed_time = current_time - start_time
            steps = (update + 1) * trainer.num_envs * trainer.unroll_length
            print(f"[LOG] Update {update+1}/{num_updates} - Steps: {steps} - Time: {elapsed_time:.2f}s - Loss: {metrics['loss']:.4f}")
            last_print_time = current_time

        if update % 5 == 0:
            print("[LOG] Running evaluation...")
            returns = trainer._evaluate_policy(trainer.network_params, num_episodes=10)
            trainer.archive.update(trainer.network_params, returns)
            print(f"[LOG] Archive Size: {len(trainer.archive)}")
            print(f"[LOG] Average returns: {returns.mean(axis=0)}")
            gc.collect()

    print("[LOG] Training complete!")
    return trainer.network_params, trainer.archive


# === Main Entry ===
def main():
    print("[DEBUG] Checking JAX backend and devices...")
    backend = xla_bridge.get_backend().platform
    devices = jax.devices()
    print(f"[DEBUG] JAX is using backend: {backend}")
    print(f"[DEBUG] Available JAX devices: {devices}")
    if backend != 'gpu':
        print("[WARNING] JAX is not using GPU! Check module loading and environment setup.")

    trainer = PGMORLTrainer(
        env_fn=MorlHopper,
        num_timesteps=200_000,
        episode_length=200,
        action_repeat=1,
        num_envs=24,
        num_eval_envs=2,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        unroll_length=10,
        batch_size=256,
        num_updates_per_batch=4,
        num_objectives=2,
        discounting=0.99,
        seed=42
    )

    try:
        trained_params, archive = custom_train(trainer)
        print(f"[LOG] Final Pareto archive size: {len(archive)}")
        with open('pgmorl_archive_final.pkl', 'wb') as f:
            pickle.dump({'params': archive.policies, 'returns': archive.returns}, f)
        print("[LOG] Archive saved to pgmorl_archive_final.pkl")
    except KeyboardInterrupt:
        print("[LOG] Interrupted. Saving partial archive...")
        with open('pgmorl_archive_partial.pkl', 'wb') as f:
            pickle.dump({'params': trainer.archive.policies, 'returns': trainer.archive.returns}, f)
        print("[LOG] Partial archive saved.")

if __name__ == "__main__":
    main()
