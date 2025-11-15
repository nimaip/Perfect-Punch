import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import argparse

# Read the data
df = pd.read_csv('data.csv', header=None,
                 names=['type', 'name', 'wrist_x', 'wrist_y', 'wrist_z',
                        'elbow_x', 'elbow_y', 'elbow_z'])

# Number of frames per sequence
n = 15
num_sequences = len(df) // n

print(f"Total rows: {len(df)}")
print(f"Number of sequences: {num_sequences}")
print(f"Unique punch types: {df['type'].unique()}")

def visualize_sequence(seq_idx, show_animation=False):
    """Visualize a single punch sequence in 3D"""
    start_idx = seq_idx * n
    end_idx = start_idx + n

    sequence = df.iloc[start_idx:end_idx]
    punch_type = sequence['type'].iloc[0]
    video_name = sequence['name'].iloc[0]

    # Extract wrist and elbow coordinates
    wrist = sequence[['wrist_x', 'wrist_y', 'wrist_z']].values
    elbow = sequence[['elbow_x', 'elbow_y', 'elbow_z']].values

    if not show_animation:
        # Static 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectories
        ax.plot(wrist[:, 0], wrist[:, 1], wrist[:, 2],
                'b-o', label='Wrist', linewidth=2, markersize=6)
        ax.plot(elbow[:, 0], elbow[:, 1], elbow[:, 2],
                'r-s', label='Elbow', linewidth=2, markersize=6)

        # Connect wrist to elbow for each frame to show arm
        for i in range(n):
            ax.plot([wrist[i, 0], elbow[i, 0]],
                   [wrist[i, 1], elbow[i, 1]],
                   [wrist[i, 2], elbow[i, 2]],
                   'gray', alpha=0.3, linewidth=1)

        # Mark start and end positions
        ax.scatter(wrist[0, 0], wrist[0, 1], wrist[0, 2],
                  c='green', s=200, marker='*', label='Start (wrist)', zorder=10)
        ax.scatter(wrist[-1, 0], wrist[-1, 1], wrist[-1, 2],
                  c='darkblue', s=200, marker='X', label='End (wrist)', zorder=10)

        # Annotate frame numbers
        for i in range(0, n, 3):  # Show every 3rd frame to avoid clutter
            ax.text(wrist[i, 0], wrist[i, 1], wrist[i, 2],
                   f'  {i}', fontsize=8, alpha=0.7)

        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z (depth)', fontsize=12)
        ax.set_title(f'{punch_type} - {video_name}\nSequence {seq_idx} (15 frames)',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    else:
        # Animated 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Set axis limits based on all data
        all_coords = np.vstack([wrist, elbow])
        ax.set_xlim(all_coords[:, 0].min() - 0.1, all_coords[:, 0].max() + 0.1)
        ax.set_ylim(all_coords[:, 1].min() - 0.1, all_coords[:, 1].max() + 0.1)
        ax.set_zlim(all_coords[:, 2].min() - 0.1, all_coords[:, 2].max() + 0.1)

        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z (depth)', fontsize=12)
        ax.set_title(f'{punch_type} - {video_name}\nFrame: 0/{n-1}',
                    fontsize=14, fontweight='bold')

        # Initialize plot elements
        wrist_line, = ax.plot([], [], [], 'b-o', label='Wrist', linewidth=2, markersize=6)
        elbow_line, = ax.plot([], [], [], 'r-s', label='Elbow', linewidth=2, markersize=6)
        arm_lines = [ax.plot([], [], [], 'gray', alpha=0.5, linewidth=2)[0] for _ in range(n)]
        current_wrist = ax.scatter([], [], [], c='cyan', s=300, marker='o',
                                   label='Current wrist', zorder=10)
        current_elbow = ax.scatter([], [], [], c='orange', s=300, marker='s',
                                   label='Current elbow', zorder=10)

        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        def init():
            wrist_line.set_data([], [])
            wrist_line.set_3d_properties([])
            elbow_line.set_data([], [])
            elbow_line.set_3d_properties([])
            for line in arm_lines:
                line.set_data([], [])
                line.set_3d_properties([])
            current_wrist._offsets3d = ([], [], [])
            current_elbow._offsets3d = ([], [], [])
            return [wrist_line, elbow_line, current_wrist, current_elbow] + arm_lines

        def update(frame):
            # Update trajectory up to current frame
            wrist_line.set_data(wrist[:frame+1, 0], wrist[:frame+1, 1])
            wrist_line.set_3d_properties(wrist[:frame+1, 2])

            elbow_line.set_data(elbow[:frame+1, 0], elbow[:frame+1, 1])
            elbow_line.set_3d_properties(elbow[:frame+1, 2])

            # Update arm connections
            for i in range(frame+1):
                arm_lines[i].set_data([wrist[i, 0], elbow[i, 0]],
                                     [wrist[i, 1], elbow[i, 1]])
                arm_lines[i].set_3d_properties([wrist[i, 2], elbow[i, 2]])

            # Update current position markers
            current_wrist._offsets3d = ([wrist[frame, 0]],
                                       [wrist[frame, 1]],
                                       [wrist[frame, 2]])
            current_elbow._offsets3d = ([elbow[frame, 0]],
                                       [elbow[frame, 1]],
                                       [elbow[frame, 2]])

            ax.set_title(f'{punch_type} - {video_name}\nFrame: {frame}/{n-1}',
                        fontsize=14, fontweight='bold')

            return [wrist_line, elbow_line, current_wrist, current_elbow] + arm_lines

        anim = FuncAnimation(fig, update, frames=n, init_func=init,
                           blit=False, interval=200, repeat=True)
        return fig, anim

def visualize_multiple_sequences(num_seqs=6, start_idx=0):
    """Visualize multiple sequences in a grid"""
    cols = 3
    rows = (num_seqs + cols - 1) // cols

    fig = plt.figure(figsize=(18, 6*rows))

    for i in range(num_seqs):
        seq_idx = start_idx + i
        if seq_idx >= num_sequences:
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
                'b-o', label='Wrist', linewidth=1.5, markersize=4)
        ax.plot(elbow[:, 0], elbow[:, 1], elbow[:, 2],
                'r-s', label='Elbow', linewidth=1.5, markersize=4)

        # Connect wrist to elbow
        for j in range(n):
            ax.plot([wrist[j, 0], elbow[j, 0]],
                   [wrist[j, 1], elbow[j, 1]],
                   [wrist[j, 2], elbow[j, 2]],
                   'gray', alpha=0.2, linewidth=0.5)

        ax.scatter(wrist[0, 0], wrist[0, 1], wrist[0, 2],
                  c='green', s=100, marker='*', zorder=10)
        ax.scatter(wrist[-1, 0], wrist[-1, 1], wrist[-1, 2],
                  c='darkblue', s=100, marker='X', zorder=10)

        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.set_title(f'{punch_type}\n(Seq {seq_idx})', fontsize=10)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    return fig

def compare_punch_types():
    """Compare different punch types side by side"""
    punch_types = df['type'].unique()

    fig = plt.figure(figsize=(18, 6))

    for idx, punch_type in enumerate(punch_types[:3]):  # Show first 3 types
        # Get first sequence of this type
        type_seqs = df[df['type'] == punch_type]
        if len(type_seqs) < n:
            continue

        sequence = type_seqs.iloc[:n]
        wrist = sequence[['wrist_x', 'wrist_y', 'wrist_z']].values
        elbow = sequence[['elbow_x', 'elbow_y', 'elbow_z']].values

        ax = fig.add_subplot(1, 3, idx+1, projection='3d')

        ax.plot(wrist[:, 0], wrist[:, 1], wrist[:, 2],
                'b-o', label='Wrist', linewidth=2, markersize=6)
        ax.plot(elbow[:, 0], elbow[:, 1], elbow[:, 2],
                'r-s', label='Elbow', linewidth=2, markersize=6)

        for i in range(n):
            ax.plot([wrist[i, 0], elbow[i, 0]],
                   [wrist[i, 1], elbow[i, 1]],
                   [wrist[i, 2], elbow[i, 2]],
                   'gray', alpha=0.3, linewidth=1)

        ax.scatter(wrist[0, 0], wrist[0, 1], wrist[0, 2],
                  c='green', s=200, marker='*', label='Start', zorder=10)
        ax.scatter(wrist[-1, 0], wrist[-1, 1], wrist[-1, 2],
                  c='darkblue', s=200, marker='X', label='End', zorder=10)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (depth)')
        ax.set_title(f'{punch_type}', fontsize=14, fontweight='bold')
        ax.legend()

    plt.suptitle('Comparison of Punch Types', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize punch sequences from data.csv')
    parser.add_argument('--sequence', type=int, default=0,
                       help='Sequence index to visualize (default: 0)')
    parser.add_argument('--animate', action='store_true',
                       help='Show animation instead of static plot')
    parser.add_argument('--multiple', type=int, default=0,
                       help='Show multiple sequences in a grid (specify number)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare different punch types')

    args = parser.parse_args()

    if args.compare:
        print("Comparing punch types...")
        fig = compare_punch_types()
        plt.show()
    elif args.multiple > 0:
        print(f"Showing {args.multiple} sequences...")
        fig = visualize_multiple_sequences(num_seqs=args.multiple, start_idx=args.sequence)
        plt.show()
    elif args.animate:
        print(f"Animating sequence {args.sequence}...")
        fig, anim = visualize_sequence(args.sequence, show_animation=True)
        plt.show()
    else:
        print(f"Visualizing sequence {args.sequence}...")
        fig = visualize_sequence(args.sequence, show_animation=False)
        plt.show()
