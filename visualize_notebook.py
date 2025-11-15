"""
Add this to a new cell in your Jupyter notebook to visualize punch sequences
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load your data
df = pd.read_csv('data.csv', header=None,
                 names=['type', 'name', 'wrist_x', 'wrist_y', 'wrist_z',
                        'elbow_x', 'elbow_y', 'elbow_z'])

n = 15  # frames per sequence

def plot_punch_sequence(seq_idx=0):
    """Plot a single punch sequence in 3D"""
    start_idx = seq_idx * n
    end_idx = start_idx + n

    sequence = df.iloc[start_idx:end_idx]
    punch_type = sequence['type'].iloc[0]
    video_name = sequence['name'].iloc[0]

    # Extract coordinates
    wrist = sequence[['wrist_x', 'wrist_y', 'wrist_z']].values
    elbow = sequence[['elbow_x', 'elbow_y', 'elbow_z']].values

    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot wrist trajectory
    ax.plot(wrist[:, 0], wrist[:, 1], wrist[:, 2],
            'b-o', label='Wrist', linewidth=2.5, markersize=8)

    # Plot elbow trajectory
    ax.plot(elbow[:, 0], elbow[:, 1], elbow[:, 2],
            'r-s', label='Elbow', linewidth=2.5, markersize=8)

    # Draw arm (connect wrist to elbow for each frame)
    for i in range(n):
        alpha = 0.2 + (i / n) * 0.5  # Fade in over time
        ax.plot([wrist[i, 0], elbow[i, 0]],
               [wrist[i, 1], elbow[i, 1]],
               [wrist[i, 2], elbow[i, 2]],
               'gray', alpha=alpha, linewidth=2)

    # Mark start and end
    ax.scatter(wrist[0, 0], wrist[0, 1], wrist[0, 2],
              c='lime', s=300, marker='*', label='Start',
              edgecolors='darkgreen', linewidths=2, zorder=10)
    ax.scatter(wrist[-1, 0], wrist[-1, 1], wrist[-1, 2],
              c='darkblue', s=300, marker='X', label='End',
              edgecolors='navy', linewidths=2, zorder=10)

    # Add frame numbers
    for i in [0, n//4, n//2, 3*n//4, n-1]:
        ax.text(wrist[i, 0], wrist[i, 1], wrist[i, 2],
               f'  F{i}', fontsize=9, alpha=0.8, fontweight='bold')

    # Labels and title
    ax.set_xlabel('X coordinate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y coordinate', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (depth)', fontsize=12, fontweight='bold')
    ax.set_title(f'{punch_type} - {video_name}\nSequence {seq_idx} ({n} frames)',
                fontsize=16, fontweight='bold', pad=20)

    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Better viewing angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()

def plot_multiple_sequences(num_seqs=6, start_idx=0):
    """Plot multiple sequences in a grid"""
    cols = 3
    rows = (num_seqs + cols - 1) // cols

    fig = plt.figure(figsize=(18, 6*rows))

    for i in range(num_seqs):
        seq_idx = start_idx + i
        if seq_idx >= len(df) // n:
            break

        start = seq_idx * n
        end = start + n
        sequence = df.iloc[start:end]
        punch_type = sequence['type'].iloc[0]

        wrist = sequence[['wrist_x', 'wrist_y', 'wrist_z']].values
        elbow = sequence[['elbow_x', 'elbow_y', 'elbow_z']].values

        ax = fig.add_subplot(rows, cols, i+1, projection='3d')

        # Plot trajectories
        ax.plot(wrist[:, 0], wrist[:, 1], wrist[:, 2],
                'b-o', label='Wrist', linewidth=1.5, markersize=5)
        ax.plot(elbow[:, 0], elbow[:, 1], elbow[:, 2],
                'r-s', label='Elbow', linewidth=1.5, markersize=5)

        # Draw arms
        for j in range(n):
            ax.plot([wrist[j, 0], elbow[j, 0]],
                   [wrist[j, 1], elbow[j, 1]],
                   [wrist[j, 2], elbow[j, 2]],
                   'gray', alpha=0.2, linewidth=1)

        # Mark start/end
        ax.scatter(wrist[0, 0], wrist[0, 1], wrist[0, 2],
                  c='lime', s=150, marker='*', zorder=10)
        ax.scatter(wrist[-1, 0], wrist[-1, 1], wrist[-1, 2],
                  c='darkblue', s=150, marker='X', zorder=10)

        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)
        ax.set_zlabel('Z', fontsize=9)
        ax.set_title(f'{punch_type}\n(Seq {seq_idx})', fontsize=11, fontweight='bold')
        ax.view_init(elev=20, azim=45)

        if i == 0:
            ax.legend(fontsize=9)

    plt.suptitle('Punch Sequence Visualization', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()

def compare_punch_types():
    """Compare one example of each punch type"""
    punch_types = df['type'].unique()
    num_types = min(3, len(punch_types))

    fig = plt.figure(figsize=(18, 6))

    for idx, punch_type in enumerate(punch_types[:num_types]):
        # Get first sequence of this type
        type_df = df[df['type'] == punch_type]
        if len(type_df) < n:
            continue

        sequence = type_df.iloc[:n]
        wrist = sequence[['wrist_x', 'wrist_y', 'wrist_z']].values
        elbow = sequence[['elbow_x', 'elbow_y', 'elbow_z']].values

        ax = fig.add_subplot(1, num_types, idx+1, projection='3d')

        # Plot trajectories
        ax.plot(wrist[:, 0], wrist[:, 1], wrist[:, 2],
                'b-o', label='Wrist', linewidth=2, markersize=6)
        ax.plot(elbow[:, 0], elbow[:, 1], elbow[:, 2],
                'r-s', label='Elbow', linewidth=2, markersize=6)

        # Draw arms
        for i in range(n):
            alpha = 0.2 + (i / n) * 0.4
            ax.plot([wrist[i, 0], elbow[i, 0]],
                   [wrist[i, 1], elbow[i, 1]],
                   [wrist[i, 2], elbow[i, 2]],
                   'gray', alpha=alpha, linewidth=1.5)

        # Mark start/end
        ax.scatter(wrist[0, 0], wrist[0, 1], wrist[0, 2],
                  c='lime', s=250, marker='*', label='Start',
                  edgecolors='darkgreen', linewidths=2, zorder=10)
        ax.scatter(wrist[-1, 0], wrist[-1, 1], wrist[-1, 2],
                  c='darkblue', s=250, marker='X', label='End',
                  edgecolors='navy', linewidths=2, zorder=10)

        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z (depth)', fontsize=10)
        ax.set_title(f'{punch_type}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.view_init(elev=20, azim=45)

    plt.suptitle('Comparison of Punch Types', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


# Example usage:
print(f"Total sequences available: {len(df) // n}")
print(f"Punch types: {df['type'].unique()}\n")

# Uncomment the function you want to run:
# plot_punch_sequence(0)  # Plot sequence 0
# plot_multiple_sequences(6)  # Plot first 6 sequences
compare_punch_types()  # Compare different punch types
