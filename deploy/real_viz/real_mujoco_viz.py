"""
Real-time MuJoCo visualization of the real Go2 robot.

Receives robot state via UDP from deploy_compliant_no_foot_xyz_sim_visual.py
and renders it in MuJoCo with force and velocity arrows.

Packet format (23 floats = 92 bytes):
    [0:12]  motor positions (SDK order: FR_hip/thigh/calf, FL, RR, RL)
    [12:16] IMU quaternion [w, x, y, z]
    [16:19] force estimate [fx, fy, fz] (body frame)
    [19:22] velocity command [vx, vy, wz]
    [22]    recording flag (1.0 = recording marker on, 0.0 = off)

Usage:
    python real_mujoco_viz.py --scene ~/unitree_mujoco/unitree_robots/go2/scene.xml

    Run this BEFORE or AFTER starting the deploy script with --sim-visual.
    The viewer waits for UDP packets and updates in real-time.
"""

import argparse
import socket
import struct
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np


# Unitree SDK motor order -> MuJoCo joint order (qpos[7:])
# SDK:    FR(0,1,2) FL(3,4,5) RR(6,7,8) RL(9,10,11)
# MuJoCo: FL(0,1,2) FR(3,4,5) RL(6,7,8) RR(9,10,11)
SDK_TO_MUJOCO = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

UDP_PORT = 9871
STANDING_HEIGHT = 0.32  # fixed base height since we don't have position data


def main():
    parser = argparse.ArgumentParser(description="Real-time MuJoCo viewer for Go2.")
    parser.add_argument("--scene", type=str, required=True,
                        help="Path to Go2 MuJoCo scene XML.")
    parser.add_argument("--port", type=int, default=UDP_PORT,
                        help=f"UDP port to listen on (default: {UDP_PORT}).")
    args = parser.parse_args()

    # Load MuJoCo model
    mj_model = mujoco.MjModel.from_xml_path(args.scene)
    mj_data = mujoco.MjData(mj_model)
    print(f"[Viz] Loaded scene: {args.scene}")
    print(f"[Viz] nq={mj_model.nq}, nv={mj_model.nv}, nu={mj_model.nu}")

    # Print joint names for verification
    print("[Viz] Joint order (qpos[7:]):")
    for i in range(mj_model.njnt):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            print(f"  joint {i}: {name}")

    # Shared state (protected by lock)
    state_lock = threading.Lock()
    state = {
        "motor_q": np.zeros(12),
        "quat": np.array([1.0, 0.0, 0.0, 0.0]),  # w, x, y, z
        "force": np.zeros(3),
        "vel_cmd": np.zeros(3),
        "recording": False,
        "received": False,
    }

    # UDP receiver thread
    def udp_receiver():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", args.port))
        sock.settimeout(0.5)
        print(f"[Viz] Listening on UDP port {args.port}")

        while True:
            try:
                data, _ = sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(data) not in (88, 92):  # 22 or 23 floats
                continue

            n_floats = len(data) // 4
            values = struct.unpack(f"{n_floats}f", data)
            with state_lock:
                state["motor_q"][:] = values[0:12]
                state["quat"][:] = values[12:16]
                state["force"][:] = values[16:19]
                state["vel_cmd"][:] = values[19:22]
                state["recording"] = (n_floats >= 23 and values[22] > 0.5)
                state["received"] = True

    recv_thread = threading.Thread(target=udp_receiver, daemon=True)
    recv_thread.start()

    # Set initial pose
    mj_data.qpos[0:3] = [0.0, 0.0, STANDING_HEIGHT]
    mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z
    mujoco.mj_forward(mj_model, mj_data)

    # Launch viewer
    base_link_id = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    print("[Viz] Viewer launched. Waiting for robot state...")

    try:
        while viewer.is_running():
            with state_lock:
                if state["received"]:
                    # Set base orientation from IMU (position fixed)
                    mj_data.qpos[0:3] = [0.0, 0.0, STANDING_HEIGHT]
                    mj_data.qpos[3:7] = state["quat"]

                    # Set joint positions: SDK motor order -> MuJoCo order
                    for sdk_idx in range(12):
                        mj_idx = SDK_TO_MUJOCO[sdk_idx]
                        mj_data.qpos[7 + mj_idx] = state["motor_q"][sdk_idx]

                    force = state["force"].copy()
                    vel_cmd = state["vel_cmd"].copy()
                    recording = state["recording"]
                else:
                    force = np.zeros(3)
                    vel_cmd = np.zeros(3)
                    recording = False

            # Forward kinematics (no physics step)
            mujoco.mj_forward(mj_model, mj_data)

            # Draw arrows
            scn = viewer._user_scn
            scn.ngeom = 0

            base_pos = mj_data.xpos[base_link_id].copy()
            base_quat = mj_data.xquat[base_link_id]
            rot_mat = np.zeros(9)
            mujoco.mju_quat2Mat(rot_mat, base_quat)
            rot_mat = rot_mat.reshape(3, 3)

            # Red arrow: force estimate (body frame -> world frame)
            force_mag = np.linalg.norm(force[:2])  # XY magnitude
            if force_mag > 0.5:
                force_world = rot_mat @ force
                arrow_scale = 0.03  # 20N -> 0.6m
                length = np.linalg.norm(force_world[:2]) * arrow_scale
                direction = force_world.copy()
                direction[2] = 0  # project to XY plane for clarity
                d_norm = np.linalg.norm(direction)
                if d_norm > 1e-6:
                    direction /= d_norm
                    start = base_pos.copy()
                    start[2] += 0.2
                    end = start + direction * length

                    if scn.ngeom < scn.maxgeom:
                        mujoco.mjv_connector(
                            scn.geoms[scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_ARROW,
                            width=0.02,
                            from_=start,
                            to=end,
                        )
                        scn.geoms[scn.ngeom].rgba[:] = [1.0, 0.2, 0.2, 0.8]
                        scn.ngeom += 1

            # Blue arrow: velocity command (body frame -> world frame)
            vel_xy_mag = np.sqrt(vel_cmd[0] ** 2 + vel_cmd[1] ** 2)
            if vel_xy_mag > 0.05:
                vel_body = np.array([vel_cmd[0], vel_cmd[1], 0.0])
                vel_world = rot_mat @ vel_body
                arrow_scale = 0.5  # 1 m/s -> 0.5m arrow
                length = vel_xy_mag * arrow_scale
                direction = vel_world.copy()
                d_norm = np.linalg.norm(direction)
                if d_norm > 1e-6:
                    direction /= d_norm
                    start = base_pos.copy()
                    start[2] += 0.25
                    end = start + direction * length

                    if scn.ngeom < scn.maxgeom:
                        mujoco.mjv_connector(
                            scn.geoms[scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_ARROW,
                            width=0.02,
                            from_=start,
                            to=end,
                        )
                        scn.geoms[scn.ngeom].rgba[:] = [0.2, 0.4, 1.0, 0.8]
                        scn.ngeom += 1

            # Green sphere: recording marker (hovering above robot)
            if recording and scn.ngeom < scn.maxgeom:
                sphere_pos = base_pos.copy()
                sphere_pos[2] += 0.50  # above force arrow
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(
                    g,
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.05, 0.05, 0.05], dtype=np.float64),
                    pos=np.array(sphere_pos, dtype=np.float64),
                    mat=np.eye(3, dtype=np.float64).flatten(),
                    rgba=np.array([0.1, 0.95, 0.2, 0.9], dtype=np.float32),
                )
                scn.ngeom += 1

            viewer.sync()
            time.sleep(0.02)  # ~50Hz refresh

    except KeyboardInterrupt:
        print("\n[Viz] Shutting down.")

    viewer.close()


if __name__ == "__main__":
    main()
