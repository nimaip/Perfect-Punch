"""
Visualize all punches of the same type overlaid on one graph
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# Load data
df = pd.read_csv('data.csv', header=None,
                 names=['type', 'name', 'wrist_x', 'wrist_y', 'wrist_z',
                        'elbow_x', 'elbow_y', 'elbow_z'])

n = 15  # frames per sequence
num_sequences = len(df) // n

def plot_all_punches_by_type(punch_type=None, show_wrist=True, show_elbow=True):
    """
    Plot all sequences of a specific punch type on the same graph

    Args:
        punch_type: The type of punch to visualize (if None, shows all unique types in separate plots)
        show_wrist: Whether to show wrist trajectories
        show_elbow: Whether to show elbow trajectories
    """
    unique_types = df['type'].unique()

    if punch_type is None:
        # Create separate plot for each punch type
        fig = plt.figure(figsize=(18, 6))

        for idx, ptype in enumerate(unique_types[:3]):
            ax = fig.add_subplot(1, 3, idx+1, projection='3d')
            _plot_punch_type(ax, ptype, show_wrist, show_elbow)

        plt.tight_layout()
    else:
        # Single plot for specified punch type
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        _plot_punch_type(ax, punch_type, show_wrist, show_elbow)
        plt.tight_layout()

    plt.show()

def _plot_punch_type(ax, punch_type, show_wrist=True, show_elbow=True):
    """Helper function to plot all sequences of one punch type"""
    # Get all sequences of this type
    type_data = df[df['type'] == punch_type]
    num_seqs = len(type_data) // n

    # Generate colors for each sequence
    wrist_colors = cm.Blues(np.linspace(0.3, 0.9, num_seqs))
    elbow_colors = cm.Reds(np.linspace(0.3, 0.9, num_seqs))

    all_wrist_coords = []
    all_elbow_coords = []

    for seq_idx in range(num_seqs):
        start = seq_idx * n
        end = start + n
        sequence = type_data.iloc[start:end]

        if len(sequence) < n:
            continue

        wrist = sequence[['wrist_x', 'wrist_y', 'wrist_z']].values
        elbow = sequence[['elbow_x', 'elbow_y', 'elbow_z']].values

        all_wrist_coords.append(wrist)
        all_elbow_coords.append(elbow)

        # Plot wrist trajectory
        if show_wrist:
            ax.plot(wrist[:, 0], wrist[:, 1], wrist[:, 2],
                   color=wrist_colors[seq_idx], alpha=0.6,
                   linewidth=1.5, label=f'Wrist {seq_idx}' if seq_idx < 3 else '')

            # Mark start and end
            ax.scatter(wrist[0, 0], wrist[0, 1], wrist[0, 2],
                      c='lime', s=50, marker='o', alpha=0.5, zorder=5)
            ax.scatter(wrist[-1, 0], wrist[-1, 1], wrist[-1, 2],
                      c='darkblue', s=50, marker='X', alpha=0.5, zorder=5)

        # Plot elbow trajectory
        if show_elbow:
            ax.plot(elbow[:, 0], elbow[:, 1], elbow[:, 2],
                   color=elbow_colors[seq_idx], alpha=0.4,
                   linewidth=1.5, linestyle='--',
                   label=f'Elbow {seq_idx}' if seq_idx < 3 else '')

    # Calculate and plot mean trajectory
    if show_wrist and all_wrist_coords:
        mean_wrist = np.mean(all_wrist_coords, axis=0)
        ax.plot(mean_wrist[:, 0], mean_wrist[:, 1], mean_wrist[:, 2],
               'b-', linewidth=4, label='Mean Wrist', zorder=10)
        ax.scatter(mean_wrist[0, 0], mean_wrist[0, 1], mean_wrist[0, 2],
                  c='lime', s=300, marker='*', edgecolors='darkgreen',
                  linewidths=2, label='Mean Start', zorder=11)
        ax.scatter(mean_wrist[-1, 0], mean_wrist[-1, 1], mean_wrist[-1, 2],
                  c='navy', s=300, marker='X', edgecolors='black',
                  linewidths=2, label='Mean End', zorder=11)

    if show_elbow and all_elbow_coords:
        mean_elbow = np.mean(all_elbow_coords, axis=0)
        ax.plot(mean_elbow[:, 0], mean_elbow[:, 1], mean_elbow[:, 2],
               'r--', linewidth=4, label='Mean Elbow', zorder=10)

    # Styling
    ax.set_xlabel('X coordinate', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y coordinate', fontsize=11, fontweight='bold')
    ax.set_zlabel('Z (depth)', fontsize=11, fontweight='bold')
    ax.set_title(f'{punch_type}\n({num_seqs} sequences overlaid)',
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=45)

    print(f"{punch_type}: {num_seqs} sequences plotted")

def plot_wrist_only_by_type():
    """Plot only wrist trajectories for all punch types"""
    unique_types = df['type'].unique()
    fig = plt.figure(figsize=(18, 6))

    for idx, ptype in enumerate(unique_types[:3]):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        _plot_punch_type(ax, ptype, show_wrist=True, show_elbow=False)

    plt.suptitle('Wrist Trajectories Only', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

def plot_variation_analysis(punch_type):
    """
    Analyze variation within a punch type
    Shows individual trajectories + mean + standard deviation
    """
    type_data = df[df['type'] == punch_type]
    num_seqs = len(type_data) // n

    fig = plt.figure(figsize=(18, 6))

    # Collect all trajectories
    all_wrist = []
    all_elbow = []

    for seq_idx in range(num_seqs):
        start = seq_idx * n
        end = start + n
        sequence = type_data.iloc[start:end]

        if len(sequence) < n:
            continue

        wrist = sequence[['wrist_x', 'wrist_y', 'wrist_z']].values
        elbow = sequence[['elbow_x', 'elbow_y', 'elbow_z']].values

        all_wrist.append(wrist)
        all_elbow.append(elbow)

    all_wrist = np.array(all_wrist)  # shape: (num_seqs, 15, 3)
    all_elbow = np.array(all_elbow)

    mean_wrist = np.mean(all_wrist, axis=0)
    std_wrist = np.std(all_wrist, axis=0)
    mean_elbow = np.mean(all_elbow, axis=0)
    std_elbow = np.std(all_elbow, axis=0)

    # Plot 1: All trajectories
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    colors = cm.Blues(np.linspace(0.3, 0.9, num_seqs))
    for i, wrist in enumerate(all_wrist):
        ax1.plot(wrist[:, 0], wrist[:, 1], wrist[:, 2],
                color=colors[i], alpha=0.5, linewidth=1)
    ax1.plot(mean_wrist[:, 0], mean_wrist[:, 1], mean_wrist[:, 2],
            'b-', linewidth=4, label='Mean')
    ax1.set_title(f'{punch_type} - All Wrist Trajectories\n({num_seqs} samples)', fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    ax1.view_init(elev=20, azim=45)

    # Plot 2: Mean trajectory with error bounds
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.plot(mean_wrist[:, 0], mean_wrist[:, 1], mean_wrist[:, 2],
            'b-o', linewidth=3, markersize=6, label='Mean Wrist')
    ax2.plot(mean_elbow[:, 0], mean_elbow[:, 1], mean_elbow[:, 2],
            'r-s', linewidth=3, markersize=6, label='Mean Elbow')

    # Draw arm at each frame
    for i in range(n):
        ax2.plot([mean_wrist[i, 0], mean_elbow[i, 0]],
                [mean_wrist[i, 1], mean_elbow[i, 1]],
                [mean_wrist[i, 2], mean_elbow[i, 2]],
                'gray', alpha=0.3, linewidth=2)

    ax2.set_title(f'{punch_type} - Mean Trajectory', fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    ax2.view_init(elev=20, azim=45)

    # Plot 3: Variation over time (distance from mean)
    ax3 = fig.add_subplot(1, 3, 3)

    # Calculate distance from mean for each sequence at each frame
    distances = np.linalg.norm(all_wrist - mean_wrist, axis=2)  # shape: (num_seqs, 15)

    # Plot individual sequences
    for i in range(num_seqs):
        ax3.plot(range(n), distances[i], alpha=0.3, color='blue', linewidth=1)

    # Plot mean and std
    mean_distance = np.mean(distances, axis=0)
    std_distance = np.std(distances, axis=0)

    ax3.plot(range(n), mean_distance, 'b-', linewidth=3, label='Mean deviation')
    ax3.fill_between(range(n),
                     mean_distance - std_distance,
                     mean_distance + std_distance,
                     alpha=0.3, color='blue', label='±1 std')

    ax3.set_xlabel('Frame', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Distance from mean trajectory', fontsize=11, fontweight='bold')
    ax3.set_title(f'{punch_type} - Trajectory Variation\n(consistency across samples)',
                 fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n{punch_type} Analysis:")
    print(f"  Number of sequences: {num_seqs}")
    print(f"  Mean deviation from mean trajectory: {mean_distance.mean():.4f}")
    print(f"  Max deviation (least consistent frame): {mean_distance.max():.4f} at frame {mean_distance.argmax()}")
    print(f"  Min deviation (most consistent frame): {mean_distance.min():.4f} at frame {mean_distance.argmin()}")


if __name__ == "__main__":
    print(f"Total sequences: {num_sequences}")
    print(f"Unique punch types: {df['type'].unique()}\n")

    # Plot all punch types (all sequences overlaid)
    print("Plotting all sequences by type...")
    plot_all_punches_by_type()

    # Or plot just wrist trajectories
    # plot_wrist_only_by_type()

    # Or analyze variation for a specific punch type
    # unique_types = df['type'].unique()
    # for ptype in unique_types[:3]:
    #     plot_variation_analysis(ptype)
