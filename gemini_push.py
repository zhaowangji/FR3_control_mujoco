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


class JointSpaceTrajectory:
    """关节空间坐标系下的线性插值轨迹"""
    def __init__(self, start_joints, end_joints, steps):
        self.start_joints = np.array(start_joints)
        self.end_joints = np.array(end_joints)
        self.steps = steps
        self.step = (self.end_joints - self.start_joints) / self.steps
        self.trajectory = self._generate_trajectory()
        self.waypoint = self.start_joints

    def _generate_trajectory(self):
        for i in range(self.steps + 1):
            yield self.start_joints + self.step * i
        yield self.end_joints

    def get_next_waypoint(self, qpos):
        if np.allclose(qpos, self.waypoint, atol=0.02):
            try:
                self.waypoint = next(self.trajectory)
                return self.waypoint
            except StopIteration:
                pass
        return self.waypoint


def main():
    model = mujoco.MjModel.from_xml_path('model/universal_robots_ur5e/scene.xml')
    data = mujoco.MjData(model)
    my_chain = ikpy.chain.Chain.from_urdf_file("model/ur5e.urdf",
                                               active_links_mask=[False, False] + [True] * 6 + [False])

    start_joints = np.array([-1.74929516e+00, -1.37547719e+00, 2.69736147e+00,
                             -1.32188428e+00, 1.39309382e+00, -1.59265359e-03])
    data.qpos[:6] = start_joints

    # 目标末端位姿
    ee_pos = [-0.1, 0.8, 0.1]
    ee_euler = [3.14, 0, 1.57]
    ref_pos = [0, 0, -1.57, -1.34, 2.65, -1.3, 1.55, 0, 0]
    ee_orientation = tf.euler.euler2mat(*ee_euler)

    joint_angles = my_chain.inverse_kinematics(ee_pos, ee_orientation, "all", initial_position=ref_pos)
    end_joints = joint_angles[2:-1]

    joint_trajectory = JointSpaceTrajectory(start_joints, end_joints, steps=100)

    force_sensor = ForceSensor(model, data, window_size=50)
    # update_interval 调至 10~20 左右，兼顾绘图流畅度与仿真速度
    plotter = ADAMSForcePlotter(update_interval=15)

    # 查找 red_box 关节 ID 用于快速索引
    box_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_box_joint")
    box_qpos_adr = model.jnt_qposadr[box_joint_id]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer_init(viewer)
        while viewer.is_running():
            waypoint = joint_trajectory.get_next_waypoint(data.qpos[:6])
            data.ctrl[:6] = waypoint

            # 1. 物理步进
            mujoco.mj_step(model, data)

            # 2. 测量物理量
            # red_box_joint 为 1-DOF 平移关节，其 qpos 即为正方体相对初始位置的移动距离 (米)
            box_displacement = data.qpos[box_qpos_adr]
            filtered_force = force_sensor.filter()

            print(filtered_force)

            # 3. 更新可视化
            plotter.update(filtered_force, box_displacement)

            viewer.sync()


if __name__ == "__main__":
    main()