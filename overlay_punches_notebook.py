"""
Add to Jupyter notebook to see all punches of the same type overlaid
Copy these functions into a notebook cell and run them
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

def overlay_all_same_type():
    """Show all sequences of each punch type overlaid"""
    unique_types = df['type'].unique()
    fig = plt.figure(figsize=(18, 6))

    for idx, punch_type in enumerate(unique_types[:3]):
        # Get all sequences of this type
        type_data = df[df['type'] == punch_type]
        num_seqs = len(type_data) // n

        ax = fig.add_subplot(1, 3, idx+1, projection='3d')

        # Generate colors - different shade for each sequence
        wrist_colors = cm.Blues(np.linspace(0.4, 1.0, num_seqs))
        elbow_colors = cm.Reds(np.linspace(0.4, 1.0, num_seqs))

        all_wrist = []
        all_elbow = []

        # Plot each sequence
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

            # Plot this sequence
            ax.plot(wrist[:, 0], wrist[:, 1], wrist[:, 2],
                   color=wrist_colors[seq_idx], alpha=0.6, linewidth=1.5)
            ax.plot(elbow[:, 0], elbow[:, 1], elbow[:, 2],
                   color=elbow_colors[seq_idx], alpha=0.3, linewidth=1.5, linestyle='--')

            # Mark start/end for each sequence
            ax.scatter(wrist[0, 0], wrist[0, 1], wrist[0, 2],
                      c='lime', s=30, alpha=0.5, zorder=5)
            ax.scatter(wrist[-1, 0], wrist[-1, 1], wrist[-1, 2],
                      c='darkblue', s=30, marker='X', alpha=0.5, zorder=5)

        # Calculate and plot MEAN trajectory (thick line)
        if all_wrist:
            mean_wrist = np.mean(all_wrist, axis=0)
            mean_elbow = np.mean(all_elbow, axis=0)

            ax.plot(mean_wrist[:, 0], mean_wrist[:, 1], mean_wrist[:, 2],
                   'b-', linewidth=4, label='Mean Wrist', zorder=10)
            ax.plot(mean_elbow[:, 0], mean_elbow[:, 1], mean_elbow[:, 2],
                   'r--', linewidth=4, label='Mean Elbow', zorder=10)

            # Highlight mean start/end
            ax.scatter(mean_wrist[0, 0], mean_wrist[0, 1], mean_wrist[0, 2],
                      c='lime', s=400, marker='*', edgecolors='darkgreen',
                      linewidths=3, label='Mean Start', zorder=11)
            ax.scatter(mean_wrist[-1, 0], mean_wrist[-1, 1], mean_wrist[-1, 2],
                      c='navy', s=400, marker='X', edgecolors='black',
                      linewidths=3, label='Mean End', zorder=11)

        ax.set_xlabel('X', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y', fontsize=10, fontweight='bold')
        ax.set_zlabel('Z (depth)', fontsize=10, fontweight='bold')
        ax.set_title(f'{punch_type}\n{num_seqs} sequences overlaid',
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)

        print(f"{punch_type}: {num_seqs} sequences")

    plt.suptitle('All Punches of Same Type Overlaid (Mean in Bold)',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

def wrist_only_overlay():
    """Show only wrist trajectories (cleaner view)"""
    unique_types = df['type'].unique()
    fig = plt.figure(figsize=(18, 6))

    for idx, punch_type in enumerate(unique_types[:3]):
        type_data = df[df['type'] == punch_type]
        num_seqs = len(type_data) // n

        ax = fig.add_subplot(1, 3, idx+1, projection='3d')

        colors = cm.viridis(np.linspace(0.2, 0.9, num_seqs))
        all_wrist = []

        for seq_idx in range(num_seqs):
            start = seq_idx * n
            end = start + n
            sequence = type_data.iloc[start:end]

            if len(sequence) < n:
                continue

            wrist = sequence[['wrist_x', 'wrist_y', 'wrist_z']].values
            all_wrist.append(wrist)

            # Lighter individual trajectories
            ax.plot(wrist[:, 0], wrist[:, 1], wrist[:, 2],
                   color=colors[seq_idx], alpha=0.5, linewidth=1.5)

        # Bold mean trajectory
        if all_wrist:
            mean_wrist = np.mean(all_wrist, axis=0)
            ax.plot(mean_wrist[:, 0], mean_wrist[:, 1], mean_wrist[:, 2],
                   'k-', linewidth=5, label='Mean', zorder=10)

            ax.scatter(mean_wrist[0, 0], mean_wrist[0, 1], mean_wrist[0, 2],
                      c='lime', s=500, marker='*', edgecolors='darkgreen',
                      linewidths=3, label='Start', zorder=11)
            ax.scatter(mean_wrist[-1, 0], mean_wrist[-1, 1], mean_wrist[-1, 2],
                      c='red', s=500, marker='X', edgecolors='darkred',
                      linewidths=3, label='End', zorder=11)

        ax.set_xlabel('X', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y', fontsize=10, fontweight='bold')
        ax.set_zlabel('Z (depth)', fontsize=10, fontweight='bold')
        ax.set_title(f'{punch_type} - Wrist Only\n{num_seqs} sequences',
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)

    plt.suptitle('Wrist Trajectories Only (Mean in Black)',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

def variation_analysis(punch_type):
    """Analyze consistency within a punch type"""
    type_data = df[df['type'] == punch_type]
    num_seqs = len(type_data) // n

    # Collect all trajectories
    all_wrist = []
    for seq_idx in range(num_seqs):
        start = seq_idx * n
        end = start + n
        sequence = type_data.iloc[start:end]
        if len(sequence) >= n:
            wrist = sequence[['wrist_x', 'wrist_y', 'wrist_z']].values
            all_wrist.append(wrist)

    all_wrist = np.array(all_wrist)  # (num_seqs, 15, 3)
    mean_wrist = np.mean(all_wrist, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Distance from mean over time
    distances = np.linalg.norm(all_wrist - mean_wrist, axis=2)

    for i in range(num_seqs):
        ax1.plot(range(n), distances[i], alpha=0.4, linewidth=1)

    mean_dist = np.mean(distances, axis=0)
    std_dist = np.std(distances, axis=0)

    ax1.plot(range(n), mean_dist, 'b-', linewidth=3, label='Mean deviation')
    ax1.fill_between(range(n),
                     mean_dist - std_dist,
                     mean_dist + std_dist,
                     alpha=0.3, label='±1 std')

    ax1.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Distance from Mean Trajectory', fontsize=12, fontweight='bold')
    ax1.set_title(f'{punch_type} - Consistency Analysis\n(Lower = More Consistent)',
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Right plot: Total variation per sequence (bar chart)
    total_variation = np.mean(distances, axis=1)
    ax2.bar(range(num_seqs), total_variation, color='steelblue', alpha=0.7)
    ax2.axhline(y=np.mean(total_variation), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(total_variation):.4f}')

    ax2.set_xlabel('Sequence Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Avg Distance from Mean', fontsize=12, fontweight='bold')
    ax2.set_title(f'{punch_type} - Variation by Sequence\n(Shows outlier sequences)',
                 fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    print(f"\n{punch_type} Consistency Metrics:")
    print(f"  Total sequences: {num_seqs}")
    print(f"  Mean deviation: {mean_dist.mean():.4f}")
    print(f"  Most consistent frame: {mean_dist.argmin()} (deviation: {mean_dist.min():.4f})")
    print(f"  Least consistent frame: {mean_dist.argmax()} (deviation: {mean_dist.max():.4f})")
    print(f"  Most typical sequence: #{np.argmin(total_variation)}")
    print(f"  Most atypical sequence: #{np.argmax(total_variation)}")


# Run the visualizations
print(f"Total sequences: {len(df) // n}")
print(f"Punch types: {df['type'].unique()}\n")

# Show all overlaid
overlay_all_same_type()

# Or show just wrists (cleaner)
# wrist_only_overlay()

# Or analyze variation for each type
# for ptype in df['type'].unique():
#     variation_analysis(ptype)
