import time
import numpy as np
import mujoco
import mujoco.viewer


def viewer_init(viewer):
    """渲染器摄像头初始化"""
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0, 0.5, 0.5]
    viewer.cam.distance = 2.2
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -30


class TeachPendant:
    """机械臂关节示教器"""
    def __init__(self, model, data, start_joints):
        self.model = model
        self.data = data
        self.target_qpos = np.array(start_joints, dtype=np.float64)
        self.selected_joint = 0  # 当前选中的关节索引 (0~5)
        self.step_rad = np.deg2rad(1.0)  # 默认每次增减 1.0 度
        
        # 获取各关节的执行器控制限位 [min, max]
        self.ctrl_ranges = self.model.actuator_ctrlrange[:6]
        
        # 关节名称映射
        self.joint_names = [
            "1. Base (基座)",
            "2. Shoulder (大臂)",
            "3. Elbow (小臂)",
            "4. Wrist 1 (腕部1)",
            "5. Wrist 2 (腕部2)",
            "6. Wrist 3 (腕部3)"
        ]

    def print_status(self):
        """在终端单行刷新示教器状态"""
        curr_deg = np.rad2deg(self.target_qpos[self.selected_joint])
        step_deg = np.rad2deg(self.step_rad)
        name = self.joint_names[self.selected_joint]
        
        # 终端 ANSI 转义码原地清行打印
        print(f"\r[示教器] 选中: {name:<16} | 目标角: {curr_deg:>7.2f}° | 步长: {step_deg:.2f}° | 状态: 正常", end="", flush=True)

    def key_callback(self, keycode):
        """GLFW 键盘事件回调"""
        # 数字键 1 ~ 6: 选择关节 (GLFW keycode: 49 ~ 54)
        if 49 <= keycode <= 54:
            self.selected_joint = keycode - 49
            self.print_status()

        # 增加角度: '=' / '+' (61) 或 Up 箭头 (265) 或 Right 箭头 (262)
        elif keycode in (61, 265, 262):
            self.target_qpos[self.selected_joint] += self.step_rad
            # 限制在物理执行器范围内
            self.target_qpos[self.selected_joint] = np.clip(
                self.target_qpos[self.selected_joint],
                self.ctrl_ranges[self.selected_joint][0],
                self.ctrl_ranges[self.selected_joint][1]
            )
            self.print_status()

        # 减小角度: '-' (45) 或 Down 箭头 (264) 或 Left 箭头 (263)
        elif keycode in (45, 264, 263):
            self.target_qpos[self.selected_joint] -= self.step_rad
            self.target_qpos[self.selected_joint] = np.clip(
                self.target_qpos[self.selected_joint],
                self.ctrl_ranges[self.selected_joint][0],
                self.ctrl_ranges[self.selected_joint][1]
            )
            self.print_status()

        # 步长粗调 / 精调: '[' 缩小步长, ']' 放大步长
        elif keycode == 91:  # '['
            self.step_rad = max(np.deg2rad(0.1), self.step_rad / 2.0)
            self.print_status()
        elif keycode == 93:  # ']'
            self.step_rad = min(np.deg2rad(10.0), self.step_rad * 2.0)
            self.print_status()


def main():
    model = mujoco.MjModel.from_xml_path('model/universal_robots_ur5e/scene.xml')
    data = mujoco.MjData(model)

    # 初始关节角度
    start_joints = np.array([-1.57, -1.34, 2.65, -1.3, 1.55, 0.0])
    data.qpos[:6] = start_joints
    data.ctrl[:6] = start_joints
    mujoco.mj_forward(model, data)

    pendant = TeachPendant(model, data, start_joints)

    print("=" * 60)
    print("【MuJoCo 关节示教器已启动】")
    print(" 键位指南:")
    print("   [1] ~ [6]      : 切换目标关节 (J1 ~ J6)")
    print("   [+] / [↑]      : 关节角 + 步长")
    print("   [-] / [↓]      : 关节角 - 步长")
    print("   [ [ ] / [ ] ]  : 缩小 / 放大微调步长 (0.1° ~ 10°)")
    print("=" * 60)
    pendant.print_status()

    # 将 key_callback 注册进 passive viewer
    with mujoco.viewer.launch_passive(model, data, key_callback=pendant.key_callback) as viewer:
        viewer_init(viewer)

        while viewer.is_running():
            step_start = time.time()

            # 驱动控制器跟踪示教器目标角度
            data.ctrl[:6] = pendant.target_qpos

            mujoco.mj_step(model, data)
            viewer.sync()

            # 维持实时物理频率
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()