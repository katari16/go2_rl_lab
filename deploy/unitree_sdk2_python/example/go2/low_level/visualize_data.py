import torch
import matplotlib.pyplot as plt
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

        # Build options list
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
            folder_name = selected[6:]  # Remove "[DIR] " prefix
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
        print("\n[Folders]")
        folder_indices = {}
        for f in folders:
            print(f"  {idx}: {f}/")
            folder_indices[idx] = f
            idx += 1

        print("\n[Files]")
        file_indices = {}
        for f in files:
            print(f"  {idx}: {f}")
            file_indices[idx] = f
            idx += 1

        if not folders and not files:
            print("  (empty)")

        print(f"\n  b: Go back")
        print(f"  q: Quit")

        choice = input("\nSelect number or command: ").strip().lower()

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

def plot_data(data_path):
    """Load and plot the selected data file."""
    print(f"\nLoading: {data_path}")
    data = torch.load(data_path, map_location="cpu")

    print(f"Available keys: {list(data.keys())}")

    t_raw = data["time"]
    dof_pos_raw = data["dof_pos"]
    des_dof_pos_raw = data["des_dof_pos"]

    has_torques = "tau_est" in data and "tau_cmd" in data

    min_len = min(t_raw.shape[0], dof_pos_raw.shape[0], des_dof_pos_raw.shape[0])
    t = t_raw[:min_len].numpy()
    dof_pos = dof_pos_raw[:min_len].numpy()
    des_dof_pos = des_dof_pos_raw[:min_len].numpy()

    if has_torques:
        tau_est = data["tau_est"][:min_len].numpy()
        tau_cmd = data["tau_cmd"][:min_len].numpy()

    joint_order = [
        "FR_hip", "FR_thigh", "FR_calf",
        "FL_hip", "FL_thigh", "FL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
        "RL_hip", "RL_thigh", "RL_calf"
    ]

    # Plot positions
    fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex=True)
    fig.suptitle(f"Joint Positions - {os.path.basename(data_path)}", fontsize=14)

    for i in range(len(joint_order)):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        ax.plot(t, dof_pos[:, i], label="actual")
        ax.plot(t, des_dof_pos[:, i], label="target", linestyle='dashed')

        ax.set_title(f"{joint_order[i]}")
        ax.set_ylabel("Position [rad]")
        ax.grid(True)
        ax.legend(loc='upper right', fontsize=8)

        if row == 3:
            ax.set_xlabel("Time [s]")

    plt.tight_layout()

    if has_torques:
        fig2, axes2 = plt.subplots(4, 3, figsize=(16, 12), sharex=True)
        fig2.suptitle(f"Joint Torques - {os.path.basename(data_path)}", fontsize=14)

        for i in range(len(joint_order)):
            row = i // 3
            col = i % 3
            ax = axes2[row, col]

            ax.plot(t, tau_est[:, i], label="tau_est (measured)")
            ax.plot(t, tau_cmd[:, i], label="tau_cmd (calculated)", linestyle='dashed')

            ax.set_title(f"{joint_order[i]}")
            ax.set_ylabel("Torque [Nm]")
            ax.grid(True)
            ax.legend(loc='upper right', fontsize=8)

            if row == 3:
                ax.set_xlabel("Time [s]")

        plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    start_dir = os.path.dirname(os.path.abspath(__file__))

    print("PACE Data Visualizer")

    if HAS_PICK:
        selected_file = interactive_file_selector_pick(start_dir)
    else:
        print("Using fallback mode (number input).\n")
        selected_file = interactive_file_selector_fallback(start_dir)

    if selected_file:
        plot_data(selected_file)
