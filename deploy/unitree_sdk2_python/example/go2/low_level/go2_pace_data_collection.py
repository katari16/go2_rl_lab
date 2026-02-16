import time
import sys
import os
from datetime import datetime
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
import unitree_legged_const as go2
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

class Custom:
    def __init__(self):
        self.Kp = 25.0
        self.Kd = 0.75
        self.dt = 0.002 

        # --- TARGET POSITIONS ---
        self._targetPos_1 = [
            0.0, 1.36, -2.65,
            0.0, 1.36, -2.65,
            -0.2, 1.36, -2.65,
            0.2, 1.36, -2.65] 
        self._targetPos_2 = [
            0.0, 0.67, -1.3,
            0.0, 0.67, -1.3,
            0.0, 0.67, -1.3,
            0.0, 0.67, -1.3]   

        # --- SINE TRAJECTORY PRE-CALCULATION ---
        self.duration_sine = 20.0
        self.num_steps = int(self.duration_sine / self.dt)
        t_vec = torch.linspace(0, self.duration_sine, steps=self.num_steps)
        self.f1 = 2.0
        f0 = 0.1
        phase_sinusoidal = 2 * torch.pi * self.f1 * t_vec
        # pace example: phase = 2 * pi * (f0 * t + ((f1 - f0) / (2 * duration)) * t ** 2)
        phase_chirp = 2 * torch.pi * (f0 * t_vec + ((self.f1 - f0) / (2 * self.duration_sine)) * t_vec ** 2)
        sine_signal = torch.sin(phase_chirp)

        directions = torch.tensor(
        [1.0, 1.0, 1.0,
        -1.0, 1.0, 1.0,
         1.0, -1.0, 1.0,
        -1.0, -1.0, 1.0])

        # bias_offset = torch.tensor(
        #     [
        #     0.0, 4.167, -6.0,
        #     0.0, 4.167, -6.0,
        #     0.0, -4.167, -6.0,
        #     0.0, -4.167, -6.0
        #     ]
        #     )
        # scale = torch.tensor([0.3, 0.3, 0.3] * 4)

        bias_offset = torch.tensor(
            [
            -0.4, 2.5, -2.95,
            -0.4, 2.5, -2.95,
            -0.4, -2.5, -2.95,
            -0.4, -2.5, -2.95
            ]
            )
        self.scale = torch.tensor([0.3, 0.35, 0.6])  # hip, thigh, calf scales

        # self.trajectory holds the absolute positions for the 20s chirp
        self.trajectory = (sine_signal.unsqueeze(-1) + bias_offset) * directions * self.scale.repeat(4)

        # --- PERCENTAGE FLAGS ---
        self.lowCmdWriteThreadPtr = None
        self.percent_1 = 0.0 
        self.percent_2 = 0.0 
        self.percent_3 = 0.0 # Bridge phase (Stand -> Start of Sine)
        self.percent_sine = 0.0 
        
        self.duration_1 = 500 
        self.duration_2 = 500 
        self.duration_3 = 500 # 1 second for the bridge
        
        self.trajectory_counter = 0
        self.startPos = [0.0] * 12
        self.firstRun = True

        # --- DATA LOGGING ---
        self.time_buffer = []
        self.dof_pos_buffer = []
        self.des_pos_buffer = []
        self.start_time = None

        self.low_cmd = unitree_go_msg_dds__LowCmd_()  
        self.low_state = None  
        self.crc = CRC()

    def Init(self):
        self.InitLowCmd()
        # create publisher
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        # create subscriber
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

        self.sc = SportClient()  
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result['name']:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)
        print("Robot Control Released. Ready for LowCmd.")

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.dt, target=self.LowCmdWrite, name="writebasiccmd"
        )
        self.lowCmdWriteThreadPtr.Start()

    def InitLowCmd(self):
        self.low_cmd.head[0]=0xFE
        self.low_cmd.head[1]=0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0

        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q = go2.PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = go2.VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg
        # print("All motor names: ", msg.motor_state[go2.LegID["FR_0"]])
        # print("IMU state: ", msg.imu_state)
        # print("Battery state: voltage: ", msg.power_v, "current: ", msg.power_a)


    def LowCmdWrite(self):
        if self.low_state is None:
            return

        if self.firstRun:
            self.start_time = time.time()
            for i in range(12):
                self.startPos[i] = self.low_state.motor_state[i].q
            self.firstRun = False

        # --- LOGGING ---
        # Only log data while the sine signal (Phase 4) is running
        # actual_q = [self.low_state.motor_state[i].q for i in range(12)]


        self.percent_1 += 1.0 / self.duration_1
        self.percent_1 = min(self.percent_1, 1.0)

        # --- PHASE 1: START TO POS 1 (Crouch) ---
        if self.percent_1 < 1.0:
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (1 - self.percent_1) * self.startPos[i] + self.percent_1 * self._targetPos_1[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0


        # --- PHASE 2: POS 1 TO POS 2 (Stand) ---
        if (self.percent_1 == 1.0) and (self.percent_2 < 1.0):
            self.percent_2 += 1.0 / self.duration_2
            self.percent_2 = min(self.percent_2, 1.0)
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (1 - self.percent_2) * self._targetPos_1[i] + self.percent_2 * self._targetPos_2[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0


        # --- PHASE 3: POS 2 TO SINE START (The Bridge) ---
        # We use trajectory[0, i] as 'targetPos_3'
        if (self.percent_1 == 1) and (self.percent_2 == 1.0) and (self.percent_3 < 1.0):
            self.percent_3 += 1.0 / self.duration_3
            self.percent_3 = min(self.percent_3, 1.0)
            for i in range(12):
                # Interpolating between Stand Pose and the very first value of the pre-calculated sine wave
                self.low_cmd.motor_cmd[i].q = (1 - self.percent_3) * self._targetPos_2[i] + self.percent_3 * self.trajectory[0, i].item()
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0

        # --- PHASE 4: SINE SIGNAL PLAYBACK ---
        if (self.percent_3 == 1.0) and (self.trajectory_counter < self.num_steps):
            self.percent_sine = float(self.trajectory_counter) / self.num_steps
            # reset the timer here.
            if self.trajectory_counter == 0:
                self.start_time = time.time()

            for i in range(12):
                self.low_cmd.motor_cmd[i].q = self.trajectory[self.trajectory_counter, i].item()
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = self.Kp
                self.low_cmd.motor_cmd[i].kd = self.Kd
                self.low_cmd.motor_cmd[i].tau = 0

            #logging and append the buffers
            actual_q = [self.low_state.motor_state[i].q for i in range(12)]
            desired_q = [self.low_cmd.motor_cmd[i].q for i in range(12)]
            self.time_buffer.append(time.time() - self.start_time)
            self.dof_pos_buffer.append(actual_q)
            self.des_pos_buffer.append(desired_q)

            self.trajectory_counter += 1

        elif self.trajectory_counter >= self.num_steps:
            self.percent_sine = 1.0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    # ChannelFactoryInitialize(0)
    custom = Custom()
    custom.Init()
    custom.Start()

    while True:        
        # Check if the process is over
        if custom.percent_sine >= 1.0:
            time.sleep(1)
            print("\nSine Signal Complete. Saving PACE data...")

            # Create dated folder (day_month_data format)
            date_folder = datetime.now().strftime("%d_%m_data")
            os.makedirs(date_folder, exist_ok=True)

            # Generate filename: chirp_{f1}hz_p{kp}d{kd}_h{hip}t{thigh}c{calf}.pt
            h, t, c = custom.scale[0].item(), custom.scale[1].item(), custom.scale[2].item()
            filename = f"chirp_{custom.f1:.1f}hz_p{custom.Kp:.0f}d{custom.Kd:.1f}_h{h:.2f}t{t:.2f}c{c:.2f}.pt"
            filepath = os.path.join(date_folder, filename)

            torch.save({
                "time": torch.tensor(custom.time_buffer, dtype=torch.float32),
                "dof_pos": torch.tensor(custom.dof_pos_buffer, dtype=torch.float32),
                "des_dof_pos": torch.tensor(custom.des_pos_buffer, dtype=torch.float32),
            }, filepath)

            print(f"File saved: {filepath}")
            print(f"Steps: {len(custom.time_buffer)}")
            sys.exit(0)

        # Terminal feedback
        if custom.percent_sine > 0:
            print(f"Phase 4 (Chirp): {custom.percent_sine*100:.1f}%", end='\r')
        elif custom.percent_3 > 0:
            print("Phase 3 (Bridging to Sine Start)...", end='\r')
        elif custom.percent_2 > 0:
            print("Phase 2 (Standing Up)...", end='\r')
        elif custom.percent_1 > 0:
            print("Phase 1 (Crouching)...", end='\r')

        time.sleep(0.1)