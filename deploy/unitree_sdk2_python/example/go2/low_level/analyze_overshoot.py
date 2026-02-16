import torch
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    from pick import pick
    HAS_PICK = True
except ImportError:
    HAS_PICK = False
    print("Note: Install 'pick' for arrow-key navigation: pip install pick")

def list_items(path):
    """List folders and .pt files in path, return sorted lists."""
    folders = []
    files = []
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            folders.append(item)
        elif item.endswith('.pt'):
            files.append(item)
    return sorted(folders), sorted(files)

def interactive_file_selector_pick(start_path):
    """Navigate folders and select a .pt file using arrow keys."""
    current_path = start_path

    while True:
        folders, files = list_items(current_path)

        options = []
        options.append(".. (go back)")
        for f in folders:
            options.append(f"[DIR] {f}")
        for f in files:
            options.append(f)
        options.append("--- QUIT ---")

        title = f"Current: {current_path}\n\nUse ↑↓ arrows, Enter to select:"
        selected, index = pick(options, title, indicator="→")

        if selected == ".. (go back)":
            parent = os.path.dirname(current_path)
            if parent and parent != current_path:
                current_path = parent
        elif selected == "--- QUIT ---":
            return None
        elif selected.startswith("[DIR] "):
            folder_name = selected[6:]
            current_path = os.path.join(current_path, folder_name)
        elif selected.endswith('.pt'):
            return os.path.join(current_path, selected)

def interactive_file_selector_fallback(start_path):
    """Fallback: Navigate folders using number input."""
    current_path = start_path

    while True:
        print(f"\n{'='*60}")
        print(f"Current: {current_path}")
        print('='*60)

        folders, files = list_items(current_path)

        idx = 0
        folder_indices = {}
        for f in folders:
            print(f"  {idx}: {f}/")
            folder_indices[idx] = f
            idx += 1

        file_indices = {}
        for f in files:
            print(f"  {idx}: {f}")
            file_indices[idx] = f
            idx += 1

        print(f"\n  b: Go back | q: Quit")
        choice = input("\nSelect: ").strip().lower()

        if choice == 'q':
            return None
        elif choice == 'b':
            parent = os.path.dirname(current_path)
            if parent and parent != current_path:
                current_path = parent
        elif choice.isdigit():
            choice_idx = int(choice)
            if choice_idx in folder_indices:
                current_path = os.path.join(current_path, folder_indices[choice_idx])
            elif choice_idx in file_indices:
                return os.path.join(current_path, file_indices[choice_idx])

def parse_filename_params(filename):
    """Extract f1, f0 from filename if possible, otherwise use defaults."""
    # Default chirp params
    f0, f1, duration = 0.1, 2.0, 20.0

    # Try to parse f1 from filename like "chirp_9.0hz_..."
    basename = os.path.basename(filename)
    if 'hz' in basename.lower():
        try:
            parts = basename.split('_')
            for p in parts:
                if 'hz' in p.lower():
                    f1 = float(p.lower().replace('hz', ''))
                    break
        except:
            pass

    return f0, f1, duration

def get_instantaneous_freq(t, f0, f1, duration):
    """Calculate instantaneous frequency for chirp signal at time t."""
    # For chirp: f(t) = f0 + (f1 - f0) * t / duration
    return f0 + (f1 - f0) * t / duration

def analyze_overshoot(data_path):
    """Analyze overshoot in the data."""
    print(f"\nLoading: {data_path}")
    data = torch.load(data_path, map_location="cpu")

    t = data["time"].numpy()
    dof_pos = data["dof_pos"].numpy()
    des_dof_pos = data["des_dof_pos"].numpy()

    # Align lengths
    min_len = min(len(t), dof_pos.shape[0], des_dof_pos.shape[0])
    t = t[:min_len]
    dof_pos = dof_pos[:min_len]
    des_dof_pos = des_dof_pos[:min_len]

    # Get chirp params from filename
    f0, f1, duration = parse_filename_params(data_path)
    print(f"Chirp params: f0={f0}Hz, f1={f1}Hz, duration={duration}s")

    joint_order = [
        "FR_hip", "FR_thigh", "FR_calf",
        "FL_hip", "FL_thigh", "FL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
        "RL_hip", "RL_thigh", "RL_calf"
    ]

    # Analyze each joint
    all_overshoots = []

    print("\n" + "="*80)
    print("OVERSHOOT ANALYSIS")
    print("="*80)
    print(f"{'Joint':<12} {'Count':>8} {'Max (rad)':>12} {'Mean (rad)':>12} {'Freq Range (Hz)':>18}")
    print("-"*80)

    for i, joint in enumerate(joint_order):
        actual = dof_pos[:, i]
        target = des_dof_pos[:, i]
        error = actual - target

        # Detect overshoots: when actual goes beyond target
        # Overshoot occurs when error and target change have same sign
        # Simpler: detect when |actual| > |target| in the direction of movement

        # Calculate target direction (derivative)
        target_diff = np.diff(target, prepend=target[0])

        # Overshoot: actual went past target
        # If target is increasing and actual > target, or target is decreasing and actual < target
        overshoot_mask = ((target_diff > 0) & (actual > target)) | ((target_diff < 0) & (actual < target))

        # Get overshoot magnitudes
        overshoot_magnitude = np.abs(error[overshoot_mask])
        overshoot_times = t[overshoot_mask]

        if len(overshoot_magnitude) > 0:
            overshoot_freqs = get_instantaneous_freq(overshoot_times, f0, f1, duration)

            max_overshoot = np.max(overshoot_magnitude)
            mean_overshoot = np.mean(overshoot_magnitude)
            freq_min, freq_max = np.min(overshoot_freqs), np.max(overshoot_freqs)

            print(f"{joint:<12} {len(overshoot_magnitude):>8} {max_overshoot:>12.4f} {mean_overshoot:>12.4f} {freq_min:>7.2f} - {freq_max:.2f}")

            for mag, freq in zip(overshoot_magnitude, overshoot_freqs):
                all_overshoots.append({
                    'joint': joint,
                    'magnitude': mag,
                    'frequency': freq
                })
        else:
            print(f"{joint:<12} {0:>8} {'N/A':>12} {'N/A':>12} {'N/A':>18}")

    print("="*80)
    print(f"Total overshoots detected: {len(all_overshoots)}")

    if len(all_overshoots) == 0:
        print("No overshoots detected!")
        return

    # Convert to arrays for plotting
    magnitudes = np.array([o['magnitude'] for o in all_overshoots])
    frequencies = np.array([o['frequency'] for o in all_overshoots])

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Overshoot Analysis - {os.path.basename(data_path)}", fontsize=14)

    # 1. Histogram of overshoot magnitudes
    ax1 = axes[0, 0]
    ax1.hist(magnitudes, bins=30, edgecolor='black', alpha=0.7)
    ax1.set_xlabel("Overshoot Magnitude (rad)")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Overshoot Magnitudes")
    ax1.axvline(np.mean(magnitudes), color='r', linestyle='--', label=f'Mean: {np.mean(magnitudes):.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Histogram of frequencies where overshoots occur
    ax2 = axes[0, 1]
    ax2.hist(frequencies, bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Count")
    ax2.set_title("Overshoots by Frequency")
    ax2.grid(True, alpha=0.3)

    # 3. Scatter: magnitude vs frequency
    ax3 = axes[1, 0]
    ax3.scatter(frequencies, magnitudes, alpha=0.5, s=10)
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Overshoot Magnitude (rad)")
    ax3.set_title("Overshoot Magnitude vs Frequency")
    ax3.grid(True, alpha=0.3)

    # 4. Per-joint overshoot count
    ax4 = axes[1, 1]
    joint_counts = {}
    for o in all_overshoots:
        joint_counts[o['joint']] = joint_counts.get(o['joint'], 0) + 1

    joints = list(joint_counts.keys())
    counts = list(joint_counts.values())
    colors = plt.cm.tab10(np.linspace(0, 1, len(joints)))
    ax4.barh(joints, counts, color=colors)
    ax4.set_xlabel("Overshoot Count")
    ax4.set_title("Overshoots per Joint")
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    start_dir = os.path.dirname(os.path.abspath(__file__))

    print("PACE Overshoot Analyzer")
    print("Analyzes overshoots: when actual position exceeds target")

    if HAS_PICK:
        selected_file = interactive_file_selector_pick(start_dir)
    else:
        print("Using fallback mode (number input).\n")
        selected_file = interactive_file_selector_fallback(start_dir)

    if selected_file:
        analyze_overshoot(selected_file)
