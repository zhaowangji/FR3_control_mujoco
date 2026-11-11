#!/usr/bin/env python3
"""UR5e 的 keti/pas 柔顺对接仿真。将本脚本放在项目根目录，与 model 同级。

Python 3.9；依赖 mujoco、numpy、scipy，显示曲线时还需 matplotlib。
    python gpt_admittance_control_push_set_misalignment.py
    python gpt_admittance_control_push_set_misalignment.py --headless --xy-mm 3 -2 --rpy-deg 1 -2 3

推进轴指向 pas，轴向压力取工装施加在工具上的反力在反推进方向的投影。
接口扳手取自 MuJoCo 的接触解。控制器移动笛卡尔位姿参考，经局部逆解交给
XML 中的关节位置伺服器。名义轴向间隙的绝对值进入设定范围（默认 6 mm）
且横向和姿态误差符合捕获门槛时启动三爪。间隙按两个配合面原点在
配合面法向上的距离计算。日志同时显示机械臂阶段、单爪力矩指令和三个爪的实测关节角。

初始偏差相对于对齐后的 keti 对接面坐标系。沿世界 +Y 直线推进。
横向及姿态修正由接触扳手驱动。
"""

import argparse
import os
import time
import xml.etree.ElementTree as ET
from itertools import permutations
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from _docking_misalignment import CaptureAssist, add_arguments, initialize_pose


START_JOINTS = [-1.74929517, -2.02082105, 2.40160303,
                -0.380781978, 1.39309381, -0.00159265359]


def require_id(model, kind, name):
    idx = mujoco.mj_name2id(model, kind, name)
    if idx < 0:
        raise ValueError(f"当前模型缺少 {name!r}，请检查 scene.xml 与 ur5e.xml")
    return idx


# 复用项目阻抗脚本中 nx_keti_1.obj / nx_pas_0.obj 三个配合面的本体外法向。
KETI_NORMALS = np.array([
    [0.414693234375, -0.000000004605, 0.909961274650],
    [-0.207346613405, 0.359134878329, 0.909961274493],
    [-0.207346620775, -0.359134872174, 0.909961275243],
])
PAS_NORMALS = np.array([
    [0.245872939160, 0.292792548059, 0.924023279788],
    [0.243732898598, -0.423617420106, 0.872434843139],
    [-0.376701548314, -0.065411063912, 0.924022367811],
])


def mating_rotation(pas_R, initial_R):
    """匹配三个相向斜面，在对称解中选择离初姿最近的工具世界姿态。"""
    candidates = []
    for order in permutations(range(3)):
        target_normals = -PAS_NORMALS[list(order)]
        u, _, vt = np.linalg.svd(target_normals.T @ KETI_NORMALS)
        local_R = u @ vt
        if np.linalg.det(local_R) > 0.0:
            target_R = pas_R @ local_R
            angle = Rotation.from_matrix(target_R @ initial_R.T).magnitude()
            candidates.append((angle, target_R))
    return min(candidates, key=lambda candidate: candidate[0])[1]


def _subtree_geoms(model, root):
    """标记本体及其所有子体上的几何体（keti 包含三个爪）。"""
    bodies = np.zeros(model.nbody, dtype=bool)
    bodies[root] = True
    for body in range(root + 1, model.nbody):
        bodies[body] = bodies[model.body_parentid[body]]
    return bodies[model.geom_bodyid]


class InterfaceWrench:
    """汇总 pas 对 keti 总成的接触力及关于 keti 原点的力矩，均用世界坐标。"""

    def __init__(self, model, tool_body, fixture_body):
        self.model = model
        self.tool_body = tool_body
        self.tool_geoms = _subtree_geoms(model, tool_body)
        self.fixture_geoms = _subtree_geoms(model, fixture_body)
        self.local_wrench = np.empty(6)

    def read(self, data):
        # 在上一步约束求解后、下一次 mj_step1 更新接触列表前调用。
        force = np.zeros(3)
        moment = np.zeros(3)
        normal_force, normal_moment = np.zeros(3), np.zeros(3)
        loaded = 0
        origin = data.xpos[self.tool_body]
        for i in range(data.ncon):
            con = data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 < 0 or g2 < 0 or con.efc_address < 0:
                continue
            if self.tool_geoms[g1] and self.fixture_geoms[g2]:
                sign = -1.0
            elif self.tool_geoms[g2] and self.fixture_geoms[g1]:
                sign = 1.0
            else:
                continue
            mujoco.mj_contactForce(self.model, data, i, self.local_wrench)
            # mj_contactForce 输出作用于 geom2 的扳手；接触基向量按行存储。
            to_world = con.frame.reshape(3, 3).T
            f = sign * (to_world @ self.local_wrench[:3])
            normal = sign * to_world[:, 0] * self.local_wrench[0]
            normal_force += normal
            normal_moment += np.cross(con.pos - origin, normal)
            force += f
            moment += np.cross(con.pos - origin, f) + sign * (to_world @ self.local_wrench[3:])
            if self.local_wrench[0] > 1e-4:
                loaded += 1
        return force, moment, normal_force, normal_moment, loaded


class ArmServo:
    """用模型中现有的六个关节位置伺服器跟踪 keti 原点的世界系位姿。"""

    def __init__(self, model, data, tool_body):
        self.model = model
        self.data = data
        self.tool_body = tool_body
        names = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
        self.actuators = np.array([
            require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names
        ])
        joints = np.array([
            require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name + "_joint") for name in names
        ])
        self.qadr = model.jnt_qposadr[joints]
        self.dofs = model.jnt_dofadr[joints]
        self.lower = np.maximum(model.jnt_range[joints, 0], model.actuator_ctrlrange[self.actuators, 0])
        self.upper = np.minimum(model.jnt_range[joints, 1], model.actuator_ctrlrange[self.actuators, 1])
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.regularizer = (0.015 ** 2) * np.eye(6)
        self.max_step = 0.5 * float(model.opt.timestep)  # 关节参考限速，rad/步
        self.qcmd = np.array(START_JOINTS, dtype=float)
        data.qpos[self.qadr] = self.qcmd
        data.ctrl[self.actuators] = self.qcmd
        mujoco.mj_forward(model, data)

    def command(self, pos_ref, R_ref):
        data = self.data
        position = data.xpos[self.tool_body]
        orientation = data.xmat[self.tool_body].reshape(3, 3)
        mujoco.mj_jac(self.model, data, self.jacp, self.jacr, position, self.tool_body)

        # 左乘姿态误差和 mj_jac 的角速度列均在世界系。用 0.15 m 长度
        # 将弧度残差换算为接口边缘的近似位移，避免平移与转动单位混用。
        radius = 0.15
        angle_error = Rotation.from_matrix(R_ref @ orientation.T).as_rotvec()
        error = np.r_[pos_ref - position, radius * angle_error]
        jac = np.vstack((self.jacp[:, self.dofs], radius * self.jacr[:, self.dofs]))
        dq = jac.T @ np.linalg.solve(jac @ jac.T + self.regularizer, error)

        # 从实测关节角展开局部逆解；受阻时限制虚拟伺服弹簧的变形。
        # 统一缩放保留解的方向，每步从上一条指令连续移向新的关节参考。
        dq *= min(1.0, 0.045 / max(np.max(np.abs(dq)), 1e-12))
        goal = np.clip(data.qpos[self.qadr] + dq, self.lower, self.upper)
        step = goal - self.qcmd
        self.qcmd += step * min(1.0, self.max_step / max(np.max(np.abs(step)), 1e-12))
        data.ctrl[self.actuators] = self.qcmd


def smoothstep(fraction):
    t = float(np.clip(fraction, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


class CompliantOffset:
    """积分 M x'' + D x' + K x = load，并限制速度和位移的模长。"""

    def __init__(self, mass, damping, stiffness, travel, speed):
        self.mass, self.damping, self.stiffness = mass, damping, stiffness
        self.travel, self.speed = travel, speed
        self.x, self.v = np.zeros(3), np.zeros(3)

    def update(self, load, dt):
        # 阻尼和弹簧用后向欧拉离散，抵近行程边界时移除向外的速度分量。
        self.v = ((self.mass * self.v + dt * (load - self.stiffness * self.x))
                  / (self.mass + dt * self.damping + dt * dt * self.stiffness))
        self.v *= min(1.0, self.speed / max(np.linalg.norm(self.v), 1e-12))
        self.x += dt * self.v
        length = np.linalg.norm(self.x)
        if length > self.travel:
            radial = self.x / length
            self.x = self.travel * radial
            self.v -= max(0.0, self.v @ radial) * radial
        return self.x


class DockingController:
    TOUCH_SPEED = 0.0015        # m/s，最后 15 mm 的搜索速度上限
    LOAD_RAMP_TIME = 1.5        # s
    AXIAL_MASS = 120.0          # kg，轴向虚拟质量
    AXIAL_DAMPING = 5000.0      # N s/m，每 5 N 压差约产生 1 mm/s 稳态速度
    PAW_TORQUE = -0.2           # N m，沿用原脚本的最终合爪力矩

    def __init__(self, model, data, preload_n, approach_speed, paw_gap=0.006,
                 xy_mm=(0.0, 0.0), rpy_deg=(0.0, 0.0, 0.0)):
        self.model, self.data = model, data
        self.dt = float(model.opt.timestep)
        self.preload, self.approach_speed = preload_n, approach_speed
        self.tool = require_id(model, mujoco.mjtObj.mjOBJ_BODY, "keti")
        fixture = require_id(model, mujoco.mjtObj.mjOBJ_BODY, "pas")
        self.capture = CaptureAssist(model, paw_gap, self.PAW_TORQUE)
        self.paws = self.capture.paws
        self.paw_qadr = model.jnt_qposadr[model.actuator_trnid[self.paws, 0]]
        self.arm = ArmServo(model, data, self.tool)
        self.interface = InterfaceWrench(model, self.tool, fixture)

        self.start = data.xpos[self.tool].copy()
        self.start_R = data.xmat[self.tool].reshape(3, 3).copy()
        self.seat = data.xpos[fixture].copy()
        self.seat_R = mating_rotation(data.xmat[fixture].reshape(3, 3), self.start_R)
        self.axis = np.array([0., 1., 0.])
        self.mating_axis = self.seat_R[:, 2]
        self.mating_tangent = np.eye(3) - np.outer(self.mating_axis, self.mating_axis)
        self.tangent = np.eye(3) - np.outer(self.axis, self.axis)
        self.start, self.start_R, _ = initialize_pose(
            model, data, self.tool, self.arm.qadr, self.arm.dofs,
            self.seat_R, self.seat, xy_mm, rpy_deg, self.arm.lower, self.arm.upper,
            actuators=self.arm.actuators,
        )
        self.stroke = float(self.axis @ (self.seat - self.start))
        if self.stroke <= 0.0:
            raise ValueError("偏置后的工具位置应位于 pas 的世界 -Y 一侧")
        self.arm.qcmd[:] = data.qpos[self.arm.qadr]

        self.lateral = CompliantOffset(mass=10.0, damping=250.0, stiffness=0.0,
                                       travel=0.020, speed=0.008)
        self.angular = CompliantOffset(mass=0.25, damping=4.0, stiffness=0.0,
                                       travel=np.deg2rad(15.0), speed=np.deg2rad(6.0))
        self.phase = "APPROACH"
        self.insertion = self.axial_ref = self.axial_speed = 0.0
        self.gap = self.stroke
        self.target_pressure = self.peak_pressure = self.pressure = 0.0
        self.force, self.guide_force, self.moment = np.zeros((3, 3))
        self.filter_gain = -np.expm1(-self.dt / 0.018)
        self.touch_time = self.lost_time = 0.0
        self.press_start = self.start_pressure = 0.0
        self.sample_interface()

    def sample_interface(self):
        # 同一次约束求解的力、接触坐标系和力臂必须配套读取。
        for warning in (mujoco.mjtWarning.mjWARN_CONTACTFULL, mujoco.mjtWarning.mjWARN_CNSTRFULL):
            if self.data.warning[warning].number:
                raise RuntimeError("MuJoCo 接触或约束缓冲区已满，无法用不完整的接触解调压")
        force, moment, normal_force, normal_moment, self.loaded = self.interface.read(self.data)
        self.force += self.filter_gain * (force - self.force)
        # 去除抵消几何导向的部分切向摩擦，轴向调压仍使用完整反力。
        self.guide_force += self.filter_gain * (0.25 * force + 0.75 * normal_force - self.guide_force)
        self.moment += self.filter_gain * (0.25 * moment + 0.75 * normal_moment - self.moment)
        self.pressure = max(0.0, -float(self.axis @ self.force))
        self.peak_pressure = max(self.peak_pressure, -float(self.axis @ force))

    def update_phase(self, now):
        loaded = self.loaded > 0
        self.touch_time = self.touch_time + self.dt if loaded else 0.0
        self.lost_time = 0.0 if loaded else self.lost_time + self.dt
        if self.phase == "APPROACH" and self.touch_time >= 0.008:
            self.phase = "PRESS"
            self.press_start = now
            self.start_pressure = min(self.pressure, self.preload)

        if self.phase == "PRESS":
            ramp = smoothstep((now - self.press_start) / self.LOAD_RAMP_TIME)
            self.target_pressure = self.start_pressure + ramp * (self.preload - self.start_pressure)
            if self.lost_time > 0.20:
                self.phase = "APPROACH"
                self.target_pressure = 0.0

    def advance_reference(self):
        # 自由空间平滑加减速。接触后同一个轴向位移/速度状态继续积分，
        # M v' + D v = F_target - pressure。过压时速度指向退出方向。
        dt = self.dt
        if self.phase == "PRESS":
            desired_speed = (self.target_pressure - self.pressure) / self.AXIAL_DAMPING
            if not self.loaded:
                desired_speed = min(desired_speed, self.TOUCH_SPEED)
            side_load = np.linalg.norm(self.tangent @ self.force)
            if side_load > max(6.0, 1.5 * self.preload):
                desired_speed = min(desired_speed, -self.TOUCH_SPEED)
            desired_speed = np.clip(desired_speed, -0.008, 0.003)
            tau = self.AXIAL_MASS / self.AXIAL_DAMPING
        else:
            slow = min(self.TOUCH_SPEED, self.approach_speed)
            far = smoothstep((self.gap - 0.015) / 0.025)
            desired_speed = slow + far * (self.approach_speed - slow)
            if self.loaded:
                desired_speed = 0.0
            tau = 0.16
        self.axial_speed += -np.expm1(-dt / tau) * (desired_speed - self.axial_speed)
        proposed = self.axial_ref + dt * self.axial_speed
        # 同时限制绝对推进深度和参考相对实测位置的超前量，卡住时消除积分累积。
        upper = min(self.stroke + 0.0015, self.insertion + 0.006)
        lower = min(upper, max(-0.010, self.insertion - 0.006))
        self.axial_ref = float(np.clip(proposed, lower, upper))
        if (proposed > upper and self.axial_speed > 0) or (proposed < lower and self.axial_speed < 0):
            self.axial_speed = 0.0

    def step(self):
        m, d = self.model, self.data
        mujoco.mj_step1(m, d)
        pos = d.xpos[self.tool]
        self.insertion = float(self.axis @ (pos - self.start))
        self.gap = float(self.mating_axis @ (self.seat - pos))
        self.update_phase(d.time)
        self.advance_reference()
        # 直线名义参考仅沿世界 Y；力和力矩积分生成有界柔顺位移。
        pos_ref = (self.start + self.axis * self.axial_ref
                   + self.lateral.update(self.tangent @ self.guide_force, self.dt))
        angle_offset = self.angular.update(self.moment, self.dt)
        rot_ref = Rotation.from_rotvec(angle_offset).as_matrix() @ self.start_R
        self.arm.command(pos_ref, rot_ref)

        rot = d.xmat[self.tool].reshape(3, 3)
        lateral = np.linalg.norm(self.mating_tangent @ (pos - self.seat))
        angle = Rotation.from_matrix(self.seat_R @ rot.T).magnitude()
        self.capture.update_paws(d, self.dt, self.gap, lateral, angle)
        mujoco.mj_step2(m, d)
        self.sample_interface()


class ForcePlot:
    """以仿真时间采样。箭头显示世界系接口反力，曲线显示沿插入轴的压力。"""

    def __init__(self, preload):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.preload = preload
        self.fig = plt.figure(figsize=(10, 4))
        self.vector = self.fig.add_subplot(121, projection="3d")
        self.vector.set(xlabel="World X (N)", ylabel="World Y (N)", zlabel="World Z (N)")
        self.vector.set_title("Contact force on tool")
        self.curve = self.fig.add_subplot(122)
        self.curve.set(xlabel="Axial travel from start (mm)", ylabel="Compression (N)")
        self.curve.grid(alpha=0.3)
        self.measured, = self.curve.plot([], [], label="Measured")
        self.target, = self.curve.plot([], [], "--", label="Target")
        self.curve.legend()
        self.travel, self.pressures, self.targets = [], [], []
        self.arrow = None
        self.fig.tight_layout()
        plt.show(block=False)

    def sample(self, ctrl):
        self.travel.append(1e3 * ctrl.insertion)
        self.pressures.append(ctrl.pressure)
        self.targets.append(ctrl.target_pressure)

    def draw(self, ctrl):
        if self.arrow is not None:
            self.arrow.remove()
        bound = max(self.preload, float(np.max(np.abs(ctrl.force)))) * 1.15
        self.vector.set(xlim=(-bound, bound), ylim=(-bound, bound), zlim=(-bound, bound))
        self.arrow = self.vector.quiver(0, 0, 0, *ctrl.force, color="tab:red")
        self.measured.set_data(self.travel, self.pressures)
        self.target.set_data(self.travel, self.targets)
        self.curve.relim()
        self.curve.autoscale_view()
        self.fig.canvas.draw_idle()


def load_scene(path):
    # 项目日志记录过默认 arena 在装配接触时耗尽。仅给本次仿真预留约束内存。
    scene = ET.parse(path).getroot()
    if scene.find("size") is not None:
        return mujoco.MjModel.from_xml_path(str(path))
    ET.SubElement(scene, "size", memory="64M")
    # 从字符串加载时，旧版 MuJoCo 也能从场景目录解析 include 和相对网格路径。
    # 加载发生在创建 viewer 线程之前，成功或异常都会恢复调用者的目录。
    original_dir = Path.cwd()
    try:
        os.chdir(path.parent)
        return mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))
    finally:
        os.chdir(original_dir)


def main():
    parser = argparse.ArgumentParser(description="UR5e 接口对接导纳控制")
    parser.add_argument("--headless", action="store_true", help="无界面运行")
    parser.add_argument("--no-plot", action="store_true", help="关闭力与位移图")
    parser.add_argument("--duration", type=float, default=40.0, help="仿真秒数，默认 40")
    parser.add_argument("--force", type=float, default=8.0, help="目标轴向预压力，2–25 N")
    parser.add_argument("--approach-speed", type=float, default=0.012,
                        help="远场接近速度，0.001–0.03 m/s")
    parser.add_argument("--paw-gap-mm", type=float, default=6,
                        help="启动合爪的名义轴向距离（mm，默认 6）；退出阈值再加 2 mm")
    add_arguments(parser)
    args = parser.parse_args()
    if not (np.isfinite(args.duration) and args.duration > 0):
        parser.error("--duration 必须是有限正数")
    if not 2.0 <= args.force <= 25.0 or not 0.001 <= args.approach_speed <= 0.03:
        parser.error("--force 或 --approach-speed 超出范围，请查看 --help")
    if not (np.isfinite(args.paw_gap_mm) and args.paw_gap_mm > 0):
        parser.error("--paw-gap-mm 必须是有限正数")
    path = Path(__file__).resolve().parent / "model/universal_robots_ur5e/scene.xml"
    model = load_scene(path)
    data = mujoco.MjData(model)
    try:
        ctrl = DockingController(model, data, args.force, args.approach_speed,
                                 args.paw_gap_mm / 1000, args.xy_mm, args.rpy_deg)
    except ValueError as error:
        parser.error(str(error))
    dt = float(model.opt.timestep)
    plot = None if args.headless or args.no_plot else ForcePlot(args.force)
    next_log = next_sample = next_draw = 0.0
    last_state = None

    def advance():
        nonlocal next_log, next_sample, last_state
        old_time = data.time
        ctrl.step()
        if not np.isfinite(data.time) or data.time <= old_time:
            raise RuntimeError("仿真时间异常或被重置，请检查 MuJoCo 警告")
        state = (ctrl.phase, ctrl.capture.active)
        if data.time >= next_log or state != last_state:
            angles = ", ".join(f"{a:.2f}" for a in np.rad2deg(data.qpos[ctrl.paw_qadr]))
            print(f"t={data.time:6.2f}s {ctrl.phase:10s}  压力={ctrl.pressure:6.2f} N  "
                  f"目标={ctrl.target_pressure:5.2f} N  名义轴向间隙={1e3 * ctrl.gap:6.2f} mm  "
                  f"合爪={'启用' if ctrl.capture.active else '关闭'}  "
                  f"单爪指令={data.ctrl[ctrl.paws[0]]:.3f} N·m  三爪角度=[{angles}]°")
            last_state, next_log = state, data.time + 1.0
        if plot is not None and data.time >= next_sample:
            plot.sample(ctrl)
            next_sample = data.time + 0.02

    if args.headless:
        while data.time < args.duration:
            advance()
    else:
        from mujoco import viewer

        paused, single_step = True, False

        def key_callback(key):
            nonlocal paused, single_step
            if key == 32:
                paused = not paused
            elif key in (ord("N"), ord("n")) and paused:
                single_step = True

        print("点击 MuJoCo 窗口，按空格运行或暂停，按 N 单步。")
        with viewer.launch_passive(model, data, key_callback=key_callback) as window:
            with window.lock():
                window.cam.lookat[:] = [-0.1, 0.45, 0.3]
                window.cam.distance, window.cam.azimuth, window.cam.elevation = 1.3, 180, -20
            while window.is_running() and data.time < args.duration:
                frame_start = time.monotonic()
                if single_step or not paused:
                    count = 1 if single_step else max(1, round(1.0 / (60.0 * dt)))
                    single_step = False
                    for _ in range(count):
                        advance()
                        if data.time >= args.duration:
                            break
                window.sync()
                if plot is not None:
                    if not plot.plt.fignum_exists(plot.fig.number):
                        plot = None
                    elif data.time >= next_draw:
                        plot.draw(ctrl)
                        next_draw = data.time + 0.10
                    if plot is not None:
                        plot.fig.canvas.flush_events()
                time.sleep(max(0.0, 1.0 / 60 - (time.monotonic() - frame_start)))
    print(f"结束 t={data.time:.3f} s  阶段={ctrl.phase}  原始轴向压力峰值={ctrl.peak_pressure:.2f} N  "
          f"当前压力={ctrl.pressure:.2f} N  名义轴向间隙={1e3 * ctrl.gap:.2f} mm")
    if plot is not None:
        plot.draw(ctrl)
        plot.plt.show(block=True)


if __name__ == "__main__":
    main()
