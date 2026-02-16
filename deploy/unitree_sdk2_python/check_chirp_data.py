import torch
import matplotlib.pyplot as plt
import os

# 1. Define the path and load the data
data_path = "/home/katari/sim_to_real_go2/unitree_sdk2_python/example/go2/low_level/chirp_data.pt"
# data_path = "/home/katari/sim_to_real_go2/unitree_sdk2_python/example/go2/low_level/data_p25d0_5/chirp_data_3_2_f1.pt"

data = torch.load(data_path, map_location="cpu")

t_raw = data["time"]
dof_pos_raw = data["dof_pos"]
des_dof_pos_raw = data["des_dof_pos"]

# 2. Align lengths (Truncation Fix to prevent broadcast errors)
min_len = min(t_raw.shape[0], dof_pos_raw.shape[0], des_dof_pos_raw.shape[0])
t = t_raw[:min_len].numpy()
dof_pos = dof_pos_raw[:min_len].numpy()
des_dof_pos = des_dof_pos_raw[:min_len].numpy()

# Exact joint order from your LegID mapping
joint_order = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf"
]

# --- ORIGINAL GRID PLOTTING (COMMENTED OUT) ---
fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex=True)

for i in range(len(joint_order)):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    
    ax.plot(t, dof_pos[:, i], label=f"{joint_order[i]} pos")
    ax.plot(t, des_dof_pos[:, i], label=f"{joint_order[i]} target", linestyle='dashed')
    
    ax.set_title(f"Joint {joint_order[i]} Trajectory")
    ax.set_ylabel("Joint position [rad]")
    ax.grid(True)
    ax.legend()
    
    if row == 3:
        ax.set_xlabel("Time [s]")

plt.tight_layout()
plt.show()
# ----------------------------------------------

# # 3. Setup Save Directory (Using the string path, not the data dictionary)
# save_dir = os.path.dirname(data_path)

# # 4. Generate and Save Individual Plots
# print(f"Saving plots to: {save_dir}")

# for i in range(len(joint_order)):
#     # Create a fresh figure for each joint
#     plt.figure(figsize=(10, 6))
    
#     # Plotting actual vs target
#     plt.plot(t, dof_pos[:, i], label=f"{joint_order[i]} pos")
#     plt.plot(t, des_dof_pos[:, i], label=f"{joint_order[i]} target", linestyle='dashed')
    
#     # Formatting
#     plt.title(f"Joint {joint_order[i]} Trajectory")
#     plt.xlabel("Time [s]")
#     plt.ylabel("Joint position [rad]")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
    
#     # Save the plot next to the data file
#     save_filename = f"plot_{joint_order[i]}.png"
#     save_path = os.path.join(save_dir, save_filename)
    
#     plt.savefig(save_path)
#     print(f"Saved: {save_filename}")
    
#     # Close the figure to free up memory (Prevents performance lag)
#     plt.close()

# print("\nAll individual plots have been saved successfully.")