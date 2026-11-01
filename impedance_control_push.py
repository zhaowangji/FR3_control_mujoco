import mujoco
import mujoco.viewer
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time


def viewer_init(viewer):
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0, 0.5, 0.5]
    viewer.cam.distance = 2.5
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -30


class ADAMSForcePlotter:
    def __init__(self, update_interval=15):
        plt.ion()
        self.fig = plt.figure(figsize=(13, 5))
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        self.ax_2d = self.fig.add_subplot(122)
        self.update_interval = update_interval
        self.frame_count = 0
        self.displacements = []
        self.forces = []
        self.line, = self.ax_2d.plot([], [], 'b-', lw=1.8, label=r'$|F_{contact}|$')
        self.ax_2d.set_title("Contact Force vs Box Displacement", fontsize=11)
        self.ax_2d.set_xlabel("Displacement $\\Delta y$ (m)", fontsize=10)
        self.ax_2d.set_ylabel("Force (N)", fontsize=10)
        self.ax_2d.grid(True, linestyle="--", alpha=0.6)
        self.ax_2d.legend(loc='upper left')
        self.fig.tight_layout()

    def update(self, force_vector, displacement):
        self.frame_count += 1
        if self.frame_count % self.update_interval != 0:
            return

        force_magnitude = float(np.linalg.norm(force_vector))
        self.displacements.append(displacement)
        self.forces.append(force_magnitude)

        self.ax_3d.clear()
        origin = np.array([0, 0, 0])
        force_direction = force_vector / force_magnitude if force_magnitude > 1e-6 else np.zeros(3)

        arrow_tip = force_direction * 1.5
        self.ax_3d.quiver(*origin, *arrow_tip, color='r', arrow_length_ratio=0)
        self.ax_3d.quiver(*arrow_tip, *(0.5 * force_direction), color='b', arrow_length_ratio=0.5)
        self.ax_3d.plot([0, arrow_tip[0]], [0, arrow_tip[1]], [-2, -2], 'g--')
        self.ax_3d.plot([0, 0], [2, 2], [0, arrow_tip[2]], 'm--')

        scaled_force = min(max(force_magnitude / 50.0, 0), 2)
        self.ax_3d.plot([-2, -2], [2, 2], [0, scaled_force], 'c-')
        self.ax_3d.text(-2, 2, scaled_force, f'Force: {force_magnitude:.1f} N', color='c')
        self.ax_3d.scatter(0, 0, 0, color='k', s=10)
        self.ax_3d.set_xlim([-2, 2])
        self.ax_3d.set_ylim([-2, 2])
        self.ax_3d.set_zlim([-2, 2])
        self.ax_3d.set_title("Force Direction Vector")

        self.line.set_data(self.displacements, self.forces)
        if len(self.displacements) > 1:
            x_min, x_max = min(self.displacements), max(self.displacements)
            y_max = max(max(self.forces), 1.0)
            dx = max(x_max - x_min, 0.05)
            self.ax_2d.set_xlim([min(0.0, x_min - 0.01), max(0.05, x_max + dx * 0.1)])
            self.ax_2d.set_ylim([-0.5, y_max * 1.15])

        plt.draw()
        plt.pause(0.001)
        self.frame_count = 0

def key_callback(keycode):
    global step_trigger, is_paused
    if keycode == 32:
        is_paused = not is_paused
        print("运行中..." if not is_paused else "已暂停")
    elif keycode in (ord('N'), ord('n')):
        step_trigger = True

class ForceSensor:
    def __init__(self, model, data, window_size=50):
        self.model = model
        self.data = data
        self.force_history = deque(maxlen=window_size)

    def filter(self):
        force_local_raw = self.data.sensordata[:3].copy() * -1
        self.force_history.append(force_local_raw)
        return np.mean(self.force_history, axis=0)


def compute_rotation_error(R_d, R_curr):
    """基于 SO(3) 旋转矩阵计算鲁棒的三维角误差向量"""
    e_rot = 0.5 * (
        np.cross(R_curr[:, 0], R_d[:, 0]) +
        np.cross(R_curr[:, 1], R_d[:, 1]) +
        np.cross(R_curr[:, 2], R_d[:, 2])
    )
    return e_rot


def compute_6d_jacobian(model, data, site_id):
    """计算末端执行器 6x6 雅可比矩阵 (前3行线速度，后3行角速度)"""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return np.vstack([jacp[:, :6], jacr[:, :6]])


class CartesianLinearTrajectory:
    """基于绝对仿真时间的笛卡尔空间匀速直线轨迹规划"""
    def __init__(self, start_pos, end_pos, duration):
        self.p_start = np.array(start_pos, dtype=float)
        self.p_end = np.array(end_pos, dtype=float)
        self.duration = duration
        self.v_nominal = (self.p_end - self.p_start) / self.duration

    def get_desired_state(self, t):
        if t <= 0:
            return self.p_start.copy(), np.zeros(3)
        elif t >= self.duration:
            return self.p_end.copy(), np.zeros(3)
        else:
            p_d = self.p_start + self.v_nominal * t
            v_d = self.v_nominal.copy()
            return p_d, v_d


class ImpedanceController:
    """基于目标阻抗动力学模型计算修正控制力"""
    def __init__(self, M, D, K, mass=1):
        self.M = M
        self.D = D
        self.K = K
        self.M_inv = np.linalg.inv(M)
        self.mass = mass * np.eye(3)
        self.x0_dotdot = np.zeros(3)

    def update(self, p_d, v_d, p_curr, v_curr, F_ext):
        pos_error = p_d - p_curr
        vel_error = v_d - v_curr
        
        # 阻抗模型推导出的末端指令力
        # F_control = (M_real * M_d^-1 - I) * F_ext + M_real * x_d_ddot + M_real * M_d^-1 * (D * e_dot + K * e)
        F_control = (self.mass @ self.M_inv - np.eye(3)) @ F_ext + self.mass @ self.x0_dotdot \
                    + self.mass @ self.M_inv @ (self.D @ vel_error + self.K @ pos_error)
        return F_control

class PeriodicDisturbanceGenerator:
    """接触触发式周期性阶跃扰动力生成器"""
    def __init__(self, interval=0.5, force_range=(-30.0, -5.0), contact_threshold=1.5):
        """
        :param interval: 外力刷新周期 (秒)
        :param force_range: 扰动力范围 (N)
        :param contact_threshold: 触发激活的最小接触力阈值 (N)
        """
        self.interval = interval
        self.force_range = force_range
        self.contact_threshold = contact_threshold
        
        self.is_triggered = False      # 是否已发生过有效接触
        self.last_update_time = -float('inf')
        self.current_force = 0.0

    def update(self, current_time, current_contact_force_mag):
        """
        :param current_time: 仿真时间 data.time
        :param current_contact_force_mag: 接触力幅值 norm(F)
        """
        # 1. 尚未接触阶段：检测是否首次突破接触力阈值
        if not self.is_triggered:
            if current_contact_force_mag >= self.contact_threshold:
                self.is_triggered = True
                self.last_update_time = current_time
                self.current_force = np.random.uniform(self.force_range[0], self.force_range[1])
                print(f"\n[Contact Triggered!] Time: {current_time:.3f}s, Initial Applied Force: {self.current_force:.2f} N")
            else:
                return 0.0

        # 2. 接触已经发生：按每 interval 秒刷新一次阻力
        if current_time - self.last_update_time >= self.interval:
            self.last_update_time = current_time
            self.current_force = np.random.uniform(self.force_range[0], self.force_range[1])
            print(f"[Disturbance Update] Time: {current_time:.3f}s | Applied Force: {self.current_force:.2f} N")

        return self.current_force

def main():
    global step_trigger, is_paused
    step_trigger = False
    is_paused = True

    model = mujoco.MjModel.from_xml_path('model/universal_robots_ur5e/scene.xml')
    data = mujoco.MjData(model)

    # 【修复 1】彻底关闭 XML 内置位置执行器的增益与阻尼，转换为纯力矩控制模式
    model.actuator_gainprm[:, 0] = 0.0
    model.actuator_biasprm[:, :] = 0.0
    if model.nu >= 6:
        data.ctrl[:6] = 0.0

    start_joints = np.array([-1.74929517e+00, -2.02082105e+00,
                                 2.40160303e+00, -3.80781978e-01,  1.39309381e+00, -1.59265359e-03])
    data.qpos[:6] = start_joints

    if model.nu >= 6:
        data.ctrl[:6] = start_joints

    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    box_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_box_joint")
    box_qpos_adr = model.jnt_qposadr[box_joint_id]

    # 末端初始位置及朝向
    init_pos = data.site_xpos[site_id].copy()
    init_R = data.site_xmat[site_id].reshape(3, 3).copy()

    start_pos = init_pos.copy()
    ee_pos = init_pos.copy()
    ee_pos[1] += 0.55  # 沿 Y 轴推进 0.5m

    trajectory = CartesianLinearTrajectory(start_pos, ee_pos, duration=5.0)

    # 阻抗参数矩阵
    impedance_controller = ImpedanceController(
        M=np.diag([2.0, 2.0, 2.0]),
        D=np.diag([100.0, 30.0, 100.0]),
        K=np.diag([300.0, 50.0, 300.0]),
        mass=21.0
    )

    force_sensor = ForceSensor(model, data, window_size=50)
    plotter = ADAMSForcePlotter(update_interval=15)

    red_box_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_box")
    force_disturbance_generator = PeriodicDisturbanceGenerator(interval=0.5, force_range=(-30.0, -5.0))

    # 控制参数
    F_push_ff = np.array([0.0, 30.0, 0.0])  # Y 方向 10N 前馈推力
    Kp_rot = 30.0                           # 姿态锁定刚度
    Kd_rot = 3.0                            # 姿态角速度阻尼
    max_torque = 50.0                       # 关节最大力矩保护

    print("\n[控制说明]：点击视窗后，按 [空格] 切换运行/暂停，按 [N] 单步运行。\n")
    print("\n修改模型后记得更改impedance_control的mass参数！\n")

    for i in range(model.nbody):
        b = model.body(i)
        print(i, b.name, b.mass, b.inertia)
    
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer_init(viewer)
        while viewer.is_running():
            if not is_paused or step_trigger:
                step_trigger = False

                # 1. 获取当前末端状态与 6D 雅可比
                J_6d = compute_6d_jacobian(model, data, site_id)
                current_pos = data.site_xpos[site_id].copy()
                current_R = data.site_xmat[site_id].reshape(3, 3).copy()

                ee_vel = J_6d @ data.qvel[:6]
                v_lin = ee_vel[:3]
                w_rot = ee_vel[3:]

                # 【修复 4】移除重复调用，并将传感器力转换到世界坐标系
                filtered_force_local = force_sensor.filter()
                f_world = current_R @ filtered_force_local
                contact_force_mag = float(np.linalg.norm(f_world))

                # 2. 动力学重力与科氏力偏置补偿
                tau_bias = data.qfrc_bias[:6].copy()

                # 3. 计算阻抗控制力
                p_d, v_d = trajectory.get_desired_state(data.time)
                F_impedance = impedance_controller.update(p_d, v_d, current_pos, v_lin, f_world)

                # 4. 组装 6 维末端扳手 (Wrench: 3维线力 + 3维姿态力矩)
                F_wrench = np.zeros(6)
                F_wrench[:3] = F_impedance + F_push_ff

                rot_error = compute_rotation_error(init_R, current_R)
                F_wrench[3:6] = Kp_rot * rot_error - Kd_rot * w_rot

                # 5. 映射到关节空间并叠加重力补偿
                tau_task = J_6d.T @ F_wrench
                tau_task = np.clip(tau_task, -max_torque, max_torque)
                data.qfrc_applied[:6] = tau_bias + tau_task

                # 6. 物理步进
                mujoco.mj_step(model, data)

                # 7. 测量与绘图
                box_displacement = data.qpos[box_qpos_adr]
                plotter.update(f_world, box_displacement)

                viewer.sync()

                print(f"Time: {data.time:.2f}s | Force: {contact_force_mag:.2f}N | F_control: {F_impedance}")

            else:
                # 暂停时仍然刷新视窗，保证界面正常响应
                viewer.sync()
                time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()