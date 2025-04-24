from brax.envs.hopper import Hopper as BaseHopper
from brax.v1.envs.env import State
import jax.numpy as jnp
import os
import tempfile

# Creating a custom MORL Hopper class by inheriting from the Base Hopper class
class MorlHopper(BaseHopper):
    def __init__(self, **kwargs):
        # Defining the XML content for the custom environment. 
        # This XML defines the physics, joints, and actuators of the Hopper model.
        xml_content = """<mujoco model="hopper">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" iterations="50" solver="PGS" timestep="0.002"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="1" condim="1" contype="1" margin="0.001" material="geom" rgba="0.8 0.6 0.4 1" solimp=".8 .8 .01" solref=".02 1"/>
    <motor ctrllimited="true" ctrlrange="-.4 .4"/>
  </default>
  <size nstack="131072"/>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="20 20 .125" type="plane" material="MatPlane"/>
    <body name="torso" pos="0 0 1.25">
      <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" ref="1.25" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 1.25" stiffness="0" type="hinge"/>
      <geom friction="0.9" fromto="0 0 1.45 0 0 1.05" name="torso_geom" size="0.05" type="capsule"/>
      <body name="thigh" pos="0 0 1.05">
        <joint axis="0 -1 0" name="thigh_joint" pos="0 0 1.05" range="-150 0" type="hinge"/>
        <geom friction="0.9" fromto="0 0 1.05 0 0 0.6" name="thigh_geom" size="0.05" type="capsule"/>
        <body name="leg" pos="0 0 0.35">
          <joint axis="0 -1 0" name="leg_joint" pos="0 0 0.6" range="-150 0" type="hinge"/>
          <geom friction="0.9" fromto="0 0 0.6 0 0 0.1" name="leg_geom" size="0.04" type="capsule"/>
          <body name="foot" pos="0.13/2 0 0.1">
            <joint axis="0 -1 0" name="foot_joint" pos="0 0 0.1" range="-45 45" type="hinge"/>
            <geom friction="2.0" fromto="-0.13 0 0.1 0.26 0 0.1" name="foot_geom" size="0.06" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="thigh_joint"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="leg_joint"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="foot_joint"/>
  </actuator>
  <material name='MatPlane' reflectance='.2' shininess='.1' specular='.5' texrepeat='1 1' texture="marble.png" texuniform='true'/>
</mujoco>
"""
        # Create a temporary XML file to store the custom environment definition
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)  # Writing XML content to file
            xml_path = f.name  # Store the path of the temporary file
        
        print(f"Using custom XML file: {xml_path}")  # Printing the file path for debugging
        
        # Remove 'xml_path' from kwargs if it's included to avoid passing it to the parent class
        if 'xml_path' in kwargs:
            del kwargs['xml_path']
        
        # Initializing the parent class with the remaining arguments
        super().__init__(**kwargs)
        
        # Setting the default MORL weights for the reward calculation (this could be adjusted later)
        self.weights = jnp.array([1.0, 0.0, 0.0])  # Default weights for forward velocity, control cost, and alive bonus
        
    def reset(self, rng) -> State:
        # Resetting the environment using the parent class's reset method
        state = super().reset(rng)
        
        # Adding a zero-initialized reward_vector to avoid errors in reward computation later
        reward_vector = jnp.zeros(3)
        
        # Adding reward_vector as part of the state metrics for tracking during training
        state = state.replace(
            metrics={**state.metrics, 'reward_vector': reward_vector}
        )
        return state  # Returning the updated state
        
    def step(self, state: State, action: jnp.ndarray) -> State:
        # Performing a step using the parent class's step method to update the state
        state = super().step(state, action)
        
        # Extracting forward velocity from the state observation (usually located at index 5)
        forward_vel = state.obs[5]
        
        # Calculating the control cost (action squared sum), encouraging efficiency
        ctrl_cost = jnp.sum(jnp.square(action))
        
        # Adding an alive bonus, incentivizing staying alive in the environment
        alive_bonus = 1.0
        
        # Defining the reward vector: [forward velocity, negative control cost, alive bonus]
        reward_vector = jnp.array([
            forward_vel,
            -ctrl_cost,
            alive_bonus
        ])
        
        # Scalar reward is computed as a weighted sum of the reward vector components
        scalar_reward = jnp.dot(self.weights, reward_vector)
        
        # Updating the state with the new reward and reward_vector in metrics for logging
        updated_metrics = {**state.metrics, 'reward_vector': reward_vector}
        
        return state.replace(
            reward=scalar_reward,  # Assigning the computed scalar reward
            metrics=updated_metrics  # Updating metrics with the new reward vector
        )
