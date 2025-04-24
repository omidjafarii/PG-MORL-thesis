"""
Script to visualize a saved PG-MORL archive
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# Make sure your working directory is correct for imports
import sys
sys.path.append('/content/drive/MyDrive/thesis')

def visualize_archive(archive_path='pgmorl_archive.pkl'):
    """
    Function to visualize the saved Pareto front from a PG-MORL archive.

    Args:
    archive_path (str): Path to the saved archive file containing the Pareto front.
    
    Visualizes the Pareto front from the archive using 2D and 3D scatter plots.
    Saves the plots in the 'results' directory.
    """
    # Load the archive from the provided path
    with open(archive_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract the returns (reward vectors) from the archive
    returns = np.array(data['returns'])
    
    # Check if the archive is empty
    if len(returns) == 0:
        print("Archive is empty. No policies to visualize.")
        return
    
    # Create the output directory to store plots if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Define the names of the objectives for labeling the axes
    obj_names = ['Forward Velocity', 'Control Cost', 'Alive Bonus']
    
    # Create 2D scatter plots for each pair of objectives (i.e., combinations of 2 objectives)
    for i in range(3):
        for j in range(i+1, 3):
            plt.figure(figsize=(8, 6))
            plt.scatter(returns[:, i], returns[:, j], c='blue', s=50)  # Scatter plot for the pair
            plt.xlabel(obj_names[i])  # X-axis label
            plt.ylabel(obj_names[j])  # Y-axis label
            plt.title(f'Pareto Front: {obj_names[i]} vs {obj_names[j]}')  # Title of the plot
            plt.grid(True)  # Enable grid for better readability
            plt.savefig(f'results/pareto_obj{i}_obj{j}.png')  # Save the plot to a file
            plt.close()  # Close the plot to free memory
    
    # Create a 3D plot for the entire Pareto front (using all 3 objectives)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(returns[:, 0], returns[:, 1], returns[:, 2], c='red', s=50)  # 3D scatter plot
    
    # Set labels for the 3D plot axes
    ax.set_xlabel(obj_names[0])
    ax.set_ylabel(obj_names[1])
    ax.set_zlabel(obj_names[2])
    ax.set_title('3D Pareto Front')  # Title of the 3D plot
    plt.savefig('results/pareto_3d.png')  # Save the 3D plot to a file
    plt.close()  # Close the plot to free memory
    
    # Print out the number of policies visualized and where the plots are saved
    print(f"Visualized {len(returns)} policies in the Pareto front.")
    print("Plots saved to the 'results' directory.")

if __name__ == "__main__":
    # Run the visualization when the script is executed
    visualize_archive()
