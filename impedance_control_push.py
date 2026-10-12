import mujoco
import mujoco.viewer
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


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
    def __init__(self, M, D, K, mass=1.0):
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


def main():
    model = mujoco.MjModel.from_xml_path('model/universal_robots_ur5e/scene.xml')
    data = mujoco.MjData(model)

    start_joints = np.array([-1.74929516e+00, -1.37547719e+00, 2.69736147e+00,
                             -1.32188428e+00, 1.39309382e+00, -1.59265359e-03])
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
    ee_pos[1] += 0.5  # 沿 Y 轴推进 0.5m

    trajectory = CartesianLinearTrajectory(start_pos, ee_pos, duration=5.0)

    # 阻抗参数矩阵
    impedance_controller = ImpedanceController(
        M=np.diag([2.0, 2.0, 2.0]),
        D=np.diag([100.0, 30.0, 100.0]),
        K=np.diag([300.0, 50.0, 300.0]),
        mass=1.0
    )

    force_sensor = ForceSensor(model, data, window_size=50)
    plotter = ADAMSForcePlotter(update_interval=15)

    # 控制参数
    F_push_ff = np.array([0.0, 10.0, 0.0])  # Y 方向 10N 前馈推力
    Kp_rot = 30.0                           # 姿态锁定刚度
    Kd_rot = 3.0                            # 姿态角速度阻尼
    max_torque = 50.0                       # 关节最大力矩保护

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer_init(viewer)
        while viewer.is_running():
            filtered_force = force_sensor.filter()

            # 1. 动力学重力与偏置补偿
            tau_bias = data.qfrc_bias[:6].copy()

            # 2. 获取当前末端状态与 6D 雅可比
            J_6d = compute_6d_jacobian(model, data, site_id)
            current_pos = data.site_xpos[site_id]
            current_R = data.site_xmat[site_id].reshape(3, 3)

            ee_vel = J_6d @ data.qvel[:6]
            v_lin = ee_vel[:3]
            w_rot = ee_vel[3:]

            # 3. 计算阻抗控制力（调用实例对象）
            p_d, v_d = trajectory.get_desired_state(data.time)
            F_impedance = impedance_controller.update(p_d, v_d, current_pos, v_lin, filtered_force)

            # 4. 组装 6 维末端扳手 (Wrench: 3维线力 + 3维姿态力矩)
            F_wrench = np.zeros(6)
            # 前三维：阻抗反馈力 + 前馈推力
            F_wrench[:3] = F_impedance + F_push_ff

            # 后三维：姿态刚度锁定（防止末端倾斜/旋转）
            rot_error = compute_rotation_error(init_R, current_R)
            F_wrench[3:6] = Kp_rot * rot_error - Kd_rot * w_rot

            # 5. 映射到关节空间
            tau_task = J_6d.T @ F_wrench
            tau_task = np.clip(tau_task, -max_torque, max_torque)

            # 施加总力矩
            data.qfrc_applied[:6] = tau_bias + tau_task

            # 消除自带位置控制器的复位力矩干扰
            if model.nu >= 6:
                data.ctrl[:6] = data.qpos[:6]

            # 物理步进
            mujoco.mj_step(model, data)

            # 测量与绘图
            box_displacement = data.qpos[box_qpos_adr]
            plotter.update(filtered_force, box_displacement)

            viewer.sync()


if __name__ == "__main__":
    main()