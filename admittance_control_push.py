import mujoco
import mujoco.viewer
import ikpy.chain
import transforms3d as tf
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


def viewer_init(viewer):
    """渲染器的摄像头视角初始化"""
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0, 0.5, 0.5]
    viewer.cam.distance = 2.5
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -30


class ADAMSForcePlotter:
    """实时可视化接触力矢量及类似ADAMS的位移-力测量曲线"""

    def __init__(self, update_interval=10):
        plt.ion()
        self.fig = plt.figure(figsize=(13, 5))
        
        # 左侧：3D 接触力方向与大小指示
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        # 右侧：2D 位移 - 接触力曲线 (ADAMS Measurement 风格)
        self.ax_2d = self.fig.add_subplot(122)

        self.update_interval = update_interval
        self.frame_count = 0

        # 改为普通列表，不限制长度，保留全流程数据
        self.displacements = []
        self.forces = []

        # 预先创建 2D 曲线对象
        self.line, = self.ax_2d.plot([], [], 'b-', lw=1.8, label=r'$|F_{contact}|$')
        self.ax_2d.set_title("Contact Force vs Box Displacement", fontsize=11)
        self.ax_2d.set_xlabel("Displacement $\\Delta y$ (m)", fontsize=10)
        self.ax_2d.set_ylabel("Force (N)", fontsize=10)
        self.ax_2d.grid(True, linestyle="--", alpha=0.6)
        self.ax_2d.legend(loc='upper left')

        self.fig.tight_layout()

    def update(self, force_vector, displacement):
        self.frame_count += 1
        
        # 仅在到达渲染周期时记录并更新，既保证全流程记录，又避免每步记录导致数组过大
        if self.frame_count % self.update_interval != 0:
            return

        force_magnitude = float(np.linalg.norm(force_vector))

        # 持久化存储数据点（不会被挤掉）
        self.displacements.append(displacement)
        self.forces.append(force_magnitude)

        # ========== 1. 更新 3D 矢量图 ==========
        self.ax_3d.clear()
        origin = np.array([0, 0, 0])
        force_direction = force_vector / force_magnitude if force_magnitude > 1e-6 else np.zeros(3)

        arrow_tip = force_direction * 1.5
        self.ax_3d.quiver(*origin, *arrow_tip, color='r', arrow_length_ratio=0)
        self.ax_3d.quiver(*arrow_tip, *(0.5 * force_direction), color='b', arrow_length_ratio=0.5)

        # 投影线
        self.ax_3d.plot([0, arrow_tip[0]], [0, arrow_tip[1]], [-2, -2], 'g--')
        self.ax_3d.plot([0, 0], [2, 2], [0, arrow_tip[2]], 'm--')

        # 力大小指示条
        scaled_force = min(max(force_magnitude / 50.0, 0), 2)
        self.ax_3d.plot([-2, -2], [2, 2], [0, scaled_force], 'c-')
        self.ax_3d.text(-2, 2, scaled_force, f'Force: {force_magnitude:.1f} N', color='c')

        self.ax_3d.scatter(0, 0, 0, color='k', s=10)
        self.ax_3d.set_xlim([-2, 2])
        self.ax_3d.set_ylim([-2, 2])
        self.ax_3d.set_zlim([-2, 2])
        self.ax_3d.set_title("Force Direction Vector")

        # ========== 2. 更新 2D 曲线图 ==========
        self.line.set_data(self.displacements, self.forces)

        # 动态自适应坐标轴范围（带边界保护）
        if len(self.displacements) > 1:
            x_min = min(self.displacements)
            x_max = max(self.displacements)
            y_max = max(max(self.forces), 1.0)

            # 防止位移静止时 x_min == x_max 导致坐标轴坍塌
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
        self.window_size = window_size
        self.force_history = deque(maxlen=window_size)

    def filter(self):
        """获取并滑动平均滤波力传感器数据"""
        # sensordata 获取力传感器输出 (Ur5e 末端安装的 force site)
        force_local_raw = self.data.sensordata[:3].copy() * -1
        self.force_history.append(force_local_raw)
        filtered_force = np.mean(self.force_history, axis=0)
        return filtered_force


class CartesianLinearTrajectory:
    """基于绝对仿真时间的笛卡尔空间匀速直线轨迹规划"""
    def __init__(self, start_pos, end_pos, duration):
        self.p_start = np.array(start_pos, dtype=float)
        self.p_end = np.array(end_pos, dtype=float)
        self.duration = duration
        
        # 匀速移动的速度向量
        self.v_nominal = (self.p_end - self.p_start) / self.duration

    def get_desired_state(self, t):
        """输入当前仿真时间 t，输出预期的笛卡尔位置与速度"""
        if t <= 0:
            return self.p_start.copy(), np.zeros(3)
        elif t >= self.duration:
            return self.p_end.copy(), np.zeros(3)
        else:
            # 匀速直线运动: x(t) = x0 + v * t
            p_d = self.p_start + self.v_nominal * t
            v_d = self.v_nominal.copy()
            return p_d, v_d

class AdmittanceController:

    def __init__(self, dt, M, D, K):
            self.dt = dt
            self.M = M
            self.D = D
            self.K = K
            self.M_inv = np.linalg.inv(M)

            # 导纳产生的偏移量位移与速度: e_x, e_v
            self.e_x = np.zeros(3)
            self.e_v = np.zeros(3)

    def update(self, F_ext):

        e_dotdot=self.M_inv@(-F_ext - self.D@self.e_v - self.K@self.e_x)
        self.e_v += e_dotdot * self.dt
        self.e_x += self.e_v * self.dt
        return self.e_x, self.e_v

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
    model = mujoco.MjModel.from_xml_path('model/universal_robots_ur5e/scene.xml')
    data = mujoco.MjData(model)
    my_chain = ikpy.chain.Chain.from_urdf_file("model/ur5e.urdf",
                                               active_links_mask=[False, False] + [True] * 6 + [False])

    start_joints = np.array([-1.74929516e+00, -1.37547719e+00, 2.69736147e+00,
                             -1.32188428e+00, 1.39309382e+00, -1.59265359e-03])
    data.qpos[:6] = start_joints

    # 目标末端位姿
    start_pos=[-0.1,0.3,0.1]
    ee_pos = [-0.1, 0.8, 0.1]
    ee_euler = [3.14, 0, 1.57]
    ref_pos = [0, 0, -1.57, -1.34, 2.65, -1.3, 1.55, 0, 0]
    ee_orientation = tf.euler.euler2mat(*ee_euler)

    joint_angles = my_chain.inverse_kinematics(ee_pos, ee_orientation, "all", initial_position=ref_pos)
    end_joints = joint_angles[2:-1]

    joint_trajectory = CartesianLinearTrajectory(start_pos, ee_pos, duration=5)

    force_sensor = ForceSensor(model, data, window_size=50)
    # update_interval 调至 10~20 左右，兼顾绘图流畅度与仿真速度
    plotter = ADAMSForcePlotter(update_interval=15)

    admittance_controller = AdmittanceController(dt=model.opt.timestep,
                                                 M=np.diag([2.0, 2.0, 2.0]), 
                                                 D=np.diag([1000.0, 40.0, 1000.0]), 
                                                 K=np.diag([100000.0, 100.0, 100000.0]))

    # 查找 red_box 关节 ID 用于快速索引
    box_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_box_joint")
    box_qpos_adr = model.jnt_qposadr[box_joint_id]

    red_box_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_box")
    force_disturbance_generator = PeriodicDisturbanceGenerator(interval=0.5, force_range=(-30.0, -5.0))
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer_init(viewer)
        while viewer.is_running():

            # 1. 期望笛卡尔轨迹
            p_d, v_d = joint_trajectory.get_desired_state(data.time)

            # 2. 测量滤波力并转为世界系
            filtered_force = force_sensor.filter()
            contact_force_mag = float(np.linalg.norm(filtered_force))

            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
            site_mat = data.site_xmat[site_id].reshape(3, 3)
            f_world = site_mat @ filtered_force

            # 3. 接触判定并计算时变外力扰动 (接触前返回 0.0，接触后周期性生成)
            f_dist_y = force_disturbance_generator.update(data.time, contact_force_mag)

            # 4. 施加扰动力到盒子 (仅在产生非零扰动时写入)
            data.xfrc_applied[red_box_id] = np.zeros(6)
            data.xfrc_applied[red_box_id][4] = f_dist_y  # Y 轴平移阻力

            # 5. 导纳控制器更新 (仅保留受力方向/利用高刚度约束)
            e_x, e_v = admittance_controller.update(f_world)

            p_modify = p_d + e_x
            now_pos = [0, 0] + list(data.qpos[:6]) + [0]
            joint_angles = my_chain.inverse_kinematics(
                target_position=p_modify,
                target_orientation=ee_orientation,
                orientation_mode="all",
                initial_position=now_pos
            )[2:-1]

            data.ctrl[:6] = joint_angles

            print(f"Time: {data.time:.2f}s | Force: {contact_force_mag:.2f}N | Disturbance: {f_dist_y:.1f}N | e_x_y: {e_x[1]:.4f}m")

            # 6. 物理步进
            mujoco.mj_step(model, data)

            # 7. 测量物理量与更新可视化
            box_displacement = data.qpos[box_qpos_adr]
            plotter.update(filtered_force, box_displacement)

            viewer.sync()


if __name__ == "__main__":
    main()