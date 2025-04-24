import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import jax
import jax.numpy as jnp
from brax.envs import create as brax_create

print("JAX devices:", jax.devices())

def run_brax_pg_morl_fast(env_name, n_iterations=5, n_episodes=3, max_steps=100, batch_size=1024):
    """Fast Brax rollout using GPU + JIT + batching."""
    # Setting up the environment and JAX functions for fast execution
    print(f"Creating batched Brax {env_name} environment with batch_size={batch_size}...")
    env = brax_create(env_name, batch_size=batch_size)

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    @jax.jit
    def rollout(rng, n_steps):
        def _rollout_body(i, carry):
            state, rng = carry
            rng, subkey = jax.random.split(rng)
            action = jax.random.uniform(subkey, shape=(batch_size, env.action_size), minval=-1, maxval=1)
            next_state = step_fn(state, action)
            return next_state, rng

        rng, subkey = jax.random.split(rng)
        init_state = reset_fn(subkey)
        final_state, _ = jax.lax.fori_loop(0, n_steps, _rollout_body, (init_state, rng))
        return final_state

    # Warming up JIT to make sure it's optimized before we do the actual work
    print("Warming up JIT...")
    rng = jax.random.PRNGKey(0)
    _ = rollout(rng, 1)

    print("Running timed rollout...")
    start_time = time.time()

    # Running the actual batched rollouts
    for i in range(n_iterations):
        for e in range(n_episodes):
            rng, key = jax.random.split(rng)
            _ = rollout(key, max_steps)

    total_time = time.time() - start_time
    total_steps = n_iterations * n_episodes * max_steps * batch_size

    print(f"🚀 Batched Brax run completed in {total_time:.2f} seconds — {total_steps:,} steps")
    return total_time, total_steps

def run_mujoco_pg_morl_simple(env_name, n_iterations=5, n_episodes=3, max_steps=100):
    """Run a simplified PG-MORL with MuJoCo for debugging."""
    # Setting up MuJoCo environment and doing some basic checks
    print(f"Creating MuJoCo {env_name} environment...")
    env = gym.make(env_name)

    print("Resetting environment...")
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    print("Taking steps...")
    start_time = time.time()

    total_steps = 0
    for i in range(n_iterations):
        for e in range(n_episodes):
            obs, _ = env.reset()
            for s in range(max_steps):
                action = env.action_space.sample()
                obs, reward, term, trunc, info = env.step(action)
                total_steps += 1
                if term or trunc:
                    obs, _ = env.reset()

    env.close()
    total_time = time.time() - start_time
    print(f"MuJoCo run completed in {total_time:.2f} seconds — {total_steps:,} steps")
    return total_time, total_steps

def compare_pg_morl(n_iterations=5, n_episodes=3, max_steps=100, batch_size=1024):
    print("\n🔍 Comparing MuJoCo vs Batched Brax performance (Steps Per Second)...")

    mujoco_envs = ["Hopper-v4", "HalfCheetah-v4"]
    brax_envs = ["hopper", "halfcheetah"]

    results = {"MuJoCo": {}, "Brax": {}}
    mujoco_sps_list = []
    brax_sps_list = []

    # Comparing the two environments for speed
    for mujoco_env, brax_env in zip(mujoco_envs, brax_envs):
        print(f"\n⚖️  Testing {mujoco_env} vs {brax_env}...")

        # Running Brax simulation and getting results
        brax_time, brax_steps = run_brax_pg_morl_fast(
            brax_env,
            n_iterations=n_iterations,
            n_episodes=n_episodes,
            max_steps=max_steps,
            batch_size=batch_size
        )
        # Running MuJoCo simulation and getting results
        mujoco_time, mujoco_steps = run_mujoco_pg_morl_simple(
            mujoco_env,
            n_iterations=n_iterations,
            n_episodes=n_episodes,
            max_steps=max_steps
        )

        brax_sps = brax_steps / brax_time
        mujoco_sps = mujoco_steps / mujoco_time

        # Display the steps per second for each environment
        print(f"✅ Brax SPS:   {brax_sps:,.0f}")
        print(f"✅ MuJoCo SPS: {mujoco_sps:,.0f}")

        results["Brax"][brax_env] = brax_sps
        results["MuJoCo"][mujoco_env] = mujoco_sps
        brax_sps_list.append(brax_sps)
        mujoco_sps_list.append(mujoco_sps)

    # Plotting the performance comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(mujoco_envs))
    width = 0.35

    ax.bar(x - width/2, mujoco_sps_list, width, label='MuJoCo (SPS)')
    ax.bar(x + width/2, brax_sps_list, width, label=f'Brax (SPS, batch={batch_size})')

    ax.set_ylabel('Steps Per Second')
    ax.set_title(f'Simulation Speed: MuJoCo vs Brax\n({n_iterations}×{n_episodes}×{max_steps} steps)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n{b}" for m, b in zip(mujoco_envs, brax_envs)])
    ax.legend()

    # Adding annotations for better visualization
    for i, v in enumerate(mujoco_sps_list):
        ax.text(i - width/2, v + v * 0.01, f"{v:,.0f}", ha='center')
    for i, v in enumerate(brax_sps_list):
        ax.text(i + width/2, v + v * 0.01, f"{v:,.0f}", ha='center')

    plt.tight_layout()
    plt.show()
    return results

# Running the comparison between MuJoCo and Brax
compare_pg_morl(n_iterations=5, n_episodes=3, max_steps=100, batch_size=1024)
