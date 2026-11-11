#!/usr/bin/env python3
"""现有 keti / pas 模型的笛卡尔力矩阻抗装配控制（不修改 XML）。

运行：conda activate zq_mujoco
      python gpt_impedance_control_push_set_misalignment.py
验证：python gpt_impedance_control_push_set_misalignment.py --headless --xy-mm 3 -2 --rpy-deg 1 -2 3

空格暂停/继续，N 单步。默认先暂停；--autostart 自动开始。
控制点取 keti 的配合面原点，而不是偏离接口 0.1 m 的 attachment_site。
控制律 tau = bias + J.T @ (K * pose_error - D * twist)。
接触后仅缓慢调整轴向弹簧平衡点以维持预压力，不放大原始力传感器输出。
模型的三个斜配合面用于规划装配位姿；这是一份已知模型的仿真控制器。
当前碰撞体在理想位姿之前约 5 mm 已经相交，故可能停在 CONTACT_HOLD：
这表示稳定保压但仍有几何残差，不代表完全贴合。无界面退出码 0=贴合，
2=运行完成但未贴合，1=异常；不要通过增大压力强行消除碰撞网格间隙。

初始偏差相对于对齐后的 keti 对接面坐标系。沿世界 +Y 直线推进。
接触后降低横向及转动刚度，并以有界的接触法向扳手补偿辅助导向。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from _docking_misalignment import CaptureAssist, add_arguments, initialize_pose



MODEL_PATH = Path(__file__).resolve().parent / "model/universal_robots_ur5e/scene.xml"
ARM_JOINTS = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
START_JOINTS = np.array([-1.74929517, -2.02082105, 2.40160303,
                         -0.380781978, 1.39309381, -0.00159265359])

# 当前 CAD 三个大斜面的外法向，分别在 keti / pas 本体坐标系内。
# 来自 nx_keti_1.obj / nx_pas_0.obj 的三角面面积加权法向；
# 三面延伸后的交点均为本体原点（偏差 < 0.1 um）。pas 网格自带倾斜，
# 因此不能用 body Z 轴代替实际接口轴，也不能只固定初始末端姿态。
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


@dataclass
class Config:
    paw_gap: float = 0.006
    push_force: float = 30.0       # N，接触后的目标轴向压力
    approach_speed: float = 0.015  # m/s，初始对齐后再接近
    force_ramp_time: float = 2.0
    axial_stiffness: float = 1200.0
    lateral_stiffness: float = 1800.0
    rotation_stiffness: float = 120.0
    linear_damping: float = 220.0
    angular_damping: float = 18.0
    force_gain: float = 0.00025    # m/(N*s)，仅用于移动弹簧平衡点
    max_reference_speed: float = 0.005
    guide_force_gain: float = 2.5
    guide_moment_gain: float = 3.0
    guide_force_limit: float = 28.0  # N，仅限制横向法向导向补偿
    guide_moment_limit: float = 6.0  # N·m
    max_force: float = 120.0       # 笛卡尔指令力的模长限制
    max_moment: float = 15.0
    force_stop: float = 10000     # 接口合力异常时终止本次仿真
    seating_tolerance: float = 0.001  # m，三个配合面的最大残差


def require_id(model, kind, name):
    idx = mujoco.mj_name2id(model, kind, name)
    if idx < 0:
        raise ValueError(f"模型中缺少 {name!r}，请检查模型与脚本是否匹配")
    return idx


def smoothstep(value):
    t = np.clip(value, 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))


def limit_norm(vector, limit):
    return vector * min(1.0, limit / max(float(np.linalg.norm(vector)), 1e-12))


def descendants(model, root):
    result = {root}
    for b in range(root + 1, model.nbody):
        if int(model.body_parentid[b]) in result:
            result.add(b)
    return result


def mating_rotation(pas_rotation, initial_rotation):
    """在三种 120 度对称装配姿态中选择离起始姿态最近的一种。"""
    candidates = []
    for order in permutations(range(3)):
        target_normals = -PAS_NORMALS[list(order)]
        u, _, vt = np.linalg.svd(target_normals.T @ KETI_NORMALS)
        local_rotation = u @ vt
        if np.linalg.det(local_rotation) < 0:
            continue  # 排除镜像对应关系
        target = pas_rotation @ local_rotation
        error = Rotation.from_matrix(target @ initial_rotation.T).magnitude()
        candidates.append((error, target, np.array(order)))
    _, rotation, order = min(candidates, key=lambda item: item[0])
    return rotation, order


class InterfaceController:
    def __init__(self, model, data, config, xy_mm=(0.0, 0.0), rpy_deg=(0.0, 0.0, 0.0)):
        self.model, self.data, self.cfg = model, data, config
        self.keti = require_id(model, mujoco.mjtObj.mjOBJ_BODY, "keti")
        self.pas = require_id(model, mujoco.mjtObj.mjOBJ_BODY, "pas")
        self.sensor = require_id(model, mujoco.mjtObj.mjOBJ_SENSOR, "force_sensor")
        self.sensor_site = int(model.sensor_objid[self.sensor])
        self.sensor_adr = int(model.sensor_adr[self.sensor])
        if (model.sensor_type[self.sensor] != mujoco.mjtSensor.mjSENS_FORCE
                or model.sensor_dim[self.sensor] != 3):
            raise ValueError("force_sensor 必须是三轴 force 传感器")
        self.joints = np.array([require_id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                                for n in ARM_JOINTS])
        self.qadr = model.jnt_qposadr[self.joints].copy()
        self.dofs = model.jnt_dofadr[self.joints].copy()
        self.torque_limits = np.full(6, 50.0)
        for i, joint in enumerate(self.joints):
            actuators = np.flatnonzero((model.actuator_trnid[:, 0] == joint)
                                      & (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT))
            if len(actuators) != 1:
                raise ValueError(f"关节 {ARM_JOINTS[i]} 应有且只有一个执行器")
            actuator = int(actuators[0])
            if model.actuator_forcelimited[actuator]:
                gear = abs(float(model.actuator_gear[actuator, 0]))
                self.torque_limits[i] = np.min(np.abs(model.actuator_forcerange[actuator])) * gear
            # 仅在内存中关闭六个机械臂位置伺服；保留夹爪执行器及所有 XML 文件。
            model.actuator_gainprm[actuator, :] = 0.0
            model.actuator_biasprm[actuator, :] = 0.0
            data.ctrl[actuator] = 0.0
        data.qpos[self.qadr] = START_JOINTS
        mujoco.mj_forward(model, data)
        self.initial_pos = data.xpos[self.keti].copy()
        self.initial_R = data.xmat[self.keti].reshape(3, 3).copy()
        self.seat_pos = data.xpos[self.pas].copy()
        self.target_R, self.plane_order = mating_rotation(
            data.xmat[self.pas].reshape(3, 3), self.initial_R)
        self.rotation_delta = Rotation.from_matrix(self.target_R @ self.initial_R.T).as_rotvec()
        self.normal = self.target_R[:, 2].copy()
        self.tangent = np.eye(3) - np.outer(self.normal, self.normal)
        self.push_axis = np.array([0., 1., 0.])
        self.push_tangent = np.eye(3) - np.outer(self.push_axis, self.push_axis)
        self.initial_axial = float(self.normal @ (self.initial_pos - self.seat_pos))
        if not -0.3 < self.initial_axial < -0.005:
            raise ValueError("起始位姿不在接口前方 5~300 mm 范围内")
        joint_bounds = model.jnt_range[self.joints]
        self.start_pos, self.offset_R, _ = initialize_pose(
            model, data, self.keti, self.qadr, self.dofs, self.target_R, self.seat_pos,
            xy_mm, rpy_deg, joint_bounds[:, 0], joint_bounds[:, 1],
        )
        self.start_axial = float(self.push_axis @ (self.start_pos - self.seat_pos))
        if self.start_axial >= 0.0:
            raise ValueError("偏置后的工具位置应位于 pas 的世界 -Y 一侧")
        self.axial_reference = self.start_axial
        self.capture = CaptureAssist(model, config.paw_gap, torque=-0.05)
        self.tool_bodies = descendants(model, self.keti)
        self.fixture_bodies = descendants(model, self.pas)
        self.geom_is_tool = np.isin(model.geom_bodyid, list(self.tool_bodies))
        self.geom_is_fixture = np.isin(model.geom_bodyid, list(self.fixture_bodies))
        self.state = "APPROACH"
        self.contact_time = None
        self.contact_ticks = 0
        self.seated_time = 0.0
        self.stable_contact_time = 0.0
        self.lost_time = 0.0
        self.filtered_force = 0.0
        self.guide_force, self.guide_moment = np.zeros((2, 3))
        self.force_target = 0.0
        self.last_tau = np.zeros(6)
        self.saturated_steps = 0
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.contact_wrench = np.zeros(6)
        self.last_contact_force_N = 0.0
        self.last_penetration_mm = 0.0
        self.metrics = {}

    def check_state(self):
        # MuJoCo 可能在 step 内自动复位非法状态，必须在步前检查、步后检查警告。
        if (not np.all(np.isfinite(self.data.qpos))
                or not np.all(np.isfinite(self.data.qvel))
                or np.linalg.norm(self.data.qvel) > 30):
            raise RuntimeError("状态非有限或关节速度异常，停止仿真")
        warning_names = [mujoco.mjtWarning(i).name for i, w in enumerate(self.data.warning) if w.number]
        if warning_names:
            raise RuntimeError("MuJoCo 警告，停止仿真: " + ", ".join(warning_names))

    def read_contacts(self):
        """仅统计接口之间的真实接触力，统一为作用在 keti 上的世界系力。"""
        force = np.zeros(3)
        normal_force, normal_moment = np.zeros((2, 3))
        count, penetration = 0, 0.0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 < 0 or g2 < 0 or con.efc_address < 0:
                continue
            if self.geom_is_tool[g1] and self.geom_is_fixture[g2]:
                sign = -1.0  # MuJoCo 的接触扳手方向为 geom1 -> geom2
            elif self.geom_is_tool[g2] and self.geom_is_fixture[g1]:
                sign = 1.0
            else:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, self.contact_wrench)
            frame = con.frame.reshape(3, 3)
            f = sign * frame.T @ self.contact_wrench[:3]
            normal = sign * frame[0] * self.contact_wrench[0]
            normal_force += normal
            normal_moment += np.cross(con.pos - self.data.xpos[self.keti], normal)
            force += f
            if self.contact_wrench[0] > 1e-4:
                count += 1
            penetration = max(penetration, -float(con.dist))
        return force, normal_force, normal_moment, count, penetration

    def step(self):
        m, d, c = self.model, self.data, self.cfg
        self.check_state()
        # mj_step1 刷新当前 qpos 的位置/速度/Jacobian；接触力取上一完整步结果，
        # 在下面调用 mj_step1 前读取，避免将旧约束力与新接触列表错配。
        f_contact, normal_force, normal_moment, contacts, penetration = self.read_contacts()
        self.last_contact_force_N = float(np.linalg.norm(f_contact))
        self.last_penetration_mm = 1000 * penetration
        # force sensor 在加速度阶段更新，连同该步的传感器姿态一并读取。
        sensor_world = (d.site_xmat[self.sensor_site].reshape(3, 3)
                        @ d.sensordata[self.sensor_adr:self.sensor_adr + 3])
        mujoco.mj_step1(m, d)
        self.check_state()
        dt = float(m.opt.timestep)
        pos = d.xpos[self.keti].copy()
        rotation = d.xmat[self.keti].reshape(3, 3).copy()
        mujoco.mj_jac(m, d, self.jacp, self.jacr, pos, self.keti)
        jac = np.vstack((self.jacp[:, self.dofs], self.jacr[:, self.dofs]))
        twist = jac @ d.qvel[self.dofs]
        axial = float(self.normal @ (pos - self.seat_pos))
        push_axial = float(self.push_axis @ (pos - self.seat_pos))
        compression = max(0.0, -float(self.push_axis @ f_contact))
        self.filtered_force += dt / (0.025 + dt) * (compression - self.filtered_force)
        angle = float(Rotation.from_matrix(self.target_R @ rotation.T).magnitude())
        lateral = float(np.linalg.norm(self.tangent @ (pos - self.seat_pos)))
        # 名义斜面残差估计：按 0.15 m 半径计入角度误差；不是网格最近点距离。
        pas_R = d.xmat[self.pas].reshape(3, 3)
        face_normals = PAS_NORMALS[self.plane_order] @ pas_R.T
        face_offsets = face_normals @ (pos - self.seat_pos)
        moving_normals = KETI_NORMALS @ rotation.T
        face_angles = np.arccos(np.clip(np.sum(-moving_normals * face_normals, axis=1), -1, 1))
        plane_residual = float(np.max(np.abs(face_offsets) + 0.15 * np.sin(face_angles)))
        if np.linalg.norm(f_contact) > c.force_stop:
            raise RuntimeError(f"接口合力超过 {c.force_stop:g} N，停止仿真，检查接触/参数")

        if self.state == "APPROACH":
            # 不再执行固定 +0.55 m 穿透轨迹。接近到真实接触后切换阻抗保压。
            self.axial_reference += c.approach_speed * dt
            self.contact_ticks = self.contact_ticks + 1 if contacts and self.filtered_force > 2 else 0
            if self.contact_ticks >= 20:
                self.state = "SEAT"
                self.contact_time = float(d.time)
                self.axial_reference = push_axial + self.filtered_force / c.axial_stiffness
        if self.state in ("SEAT", "HOLD", "CONTACT_HOLD"):
            ramp = smoothstep((d.time - self.contact_time) / c.force_ramp_time)
            self.force_target = 2.0 + (c.push_force - 2.0) * ramp
            rate = np.clip(c.force_gain * (self.force_target - self.filtered_force),
                           -c.max_reference_speed, c.max_reference_speed)
            # 有界的弹簧平衡点积分，不能在卡住时无限积累位移和压力。
            self.axial_reference += rate * dt
            self.axial_reference = float(np.clip(self.axial_reference,
                push_axial - 0.01, push_axial + c.push_force / c.axial_stiffness + 0.01))
            self.lost_time = self.lost_time + dt if not contacts else 0.0
            if self.lost_time > 2.0:
                self.state = "APPROACH"
                self.contact_ticks = 0
                self.axial_reference = min(self.axial_reference, push_axial + 0.002)
                self.force_target = 0.0
            is_seated = (plane_residual < c.seating_tolerance and lateral < 0.001
                         and angle < np.deg2rad(0.5) and contacts > 0
                         and abs(self.filtered_force - c.push_force) < 0.2 * c.push_force
                         and np.linalg.norm(twist[:3]) < 0.002)
            self.seated_time = self.seated_time + dt if is_seated else 0.0
            stable = (contacts > 0 and abs(self.filtered_force - c.push_force) < 0.1 * c.push_force
                      and np.linalg.norm(twist[:3]) < 0.0005
                      and np.linalg.norm(twist[3:]) < 0.005)
            self.stable_contact_time = self.stable_contact_time + dt if stable else 0.0
            if self.seated_time >= 0.5:
                self.state = "HOLD"
            elif self.stable_contact_time >= 1.0 and self.capture.active and not is_seated:
                self.state = "CONTACT_HOLD"
            elif self.state in ("HOLD", "CONTACT_HOLD"):
                self.state = "SEAT"
        self.axial_reference = min(self.axial_reference, c.push_force / c.axial_stiffness + 0.015)
        desired_pos = self.start_pos + self.push_axis * (self.axial_reference - self.start_axial)
        position_error = desired_pos - pos
        lateral_error = self.push_tangent @ position_error
        rot_error = Rotation.from_matrix(self.offset_R @ rotation.T).as_rotvec()
        # 横向及转动在接触后使用自由阻尼，只有超出柔顺行程才恢复弹簧。
        # 保留最初偏置作为行程中心，控制律不使用已知对齐位姿作为回正目标。
        if self.contact_time is not None:
            lateral_error -= limit_norm(lateral_error, 0.020)
            rot_error -= limit_norm(rot_error, np.deg2rad(15.0))
        force = (c.lateral_stiffness * lateral_error
                 + c.axial_stiffness * self.push_axis * (self.push_axis @ position_error)
                 - c.linear_damping * twist[:3])
        moment = c.rotation_stiffness * rot_error - c.angular_damping * twist[3:]
        # 有界法向导向补偿克服卡滞；无接触时衰减，合爪时退出。
        guide_on = self.contact_time is not None and not self.capture.active
        alpha = dt / (0.035 + dt)
        guide_force = c.guide_force_gain * (self.push_tangent @ normal_force) if guide_on else np.zeros(3)
        guide_moment = c.guide_moment_gain * normal_moment if guide_on else np.zeros(3)
        self.guide_force += alpha * (limit_norm(guide_force, c.guide_force_limit) - self.guide_force)
        self.guide_moment += alpha * (limit_norm(guide_moment, c.guide_moment_limit) - self.guide_moment)
        force += self.guide_force
        moment += self.guide_moment
        wrench = np.r_[limit_norm(force, c.max_force), limit_norm(moment, c.max_moment)]
        tau = d.qfrc_bias[self.dofs] + jac.T @ wrench
        self.saturated_steps += int(np.any(np.abs(tau) > self.torque_limits))
        tau = np.clip(tau, -self.torque_limits, self.torque_limits)
        # 限制的是包括重力/科氏补偿在内的最终力矩，而不是仅 task 分量。
        max_delta = 1000.0 * dt
        self.last_tau += np.clip(tau - self.last_tau, -max_delta, max_delta)
        d.qfrc_applied[self.dofs] = self.last_tau
        self.capture.update_paws(d, dt, -axial, lateral, angle)
        # 力传感器只用于诊断，其中还含惯性力；接触压力由 read_contacts 单独计算。
        self.metrics = dict(time_s=float(d.time), phase=self.state,
            paw_active=self.capture.active, paw_torque_Nm=float(d.ctrl[self.capture.paws[0]]),
            force_N=self.filtered_force, force_target_N=self.force_target,
            contact_force_N=float(np.linalg.norm(f_contact)), contacts=contacts,
            axial_offset_mm=1000 * axial, lateral_error_mm=1000 * lateral,
            angle_error_deg=float(np.rad2deg(angle)), plane_residual_mm=1000 * plane_residual,
            penetration_mm=1000 * penetration, speed_m_s=float(np.linalg.norm(twist[:3])),
            sensor_force_N=float(np.linalg.norm(sensor_world)),
            torque_ratio=float(np.max(np.abs(self.last_tau) / self.torque_limits)))
        mujoco.mj_step2(m, d)
        self.check_state()
        return self.metrics.copy()


def run(args):
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    config = Config(push_force=args.force, approach_speed=args.approach_speed,
                    paw_gap=args.paw_gap_mm / 1000)
    controller = InterfaceController(model, data, config, args.xy_mm, args.rpy_deg)
    print(f"MuJoCo {mujoco.__version__} | 模型: {MODEL_PATH}")
    print(f"接口轴: {controller.normal.round(6)} | 对齐基准相对原始姿态修正: "
          f"{np.rad2deg(np.linalg.norm(controller.rotation_delta)):.3f} deg")
    print("阶段: APPROACH 带偏差接近 → SEAT 贴合 → HOLD 保压")
    print("CONTACT_HOLD = 合爪指令已启用且稳定保压，尚未达到名义几何贴合容差。")
    samples, peaks = [], dict(force_N=0.0, penetration_mm=0.0, joint_speed_rad_s=0.0)
    paused = not args.autostart
    single_step = False
    next_log, next_sample = 0.0, 0.0
    started = time.monotonic()
    failure, failure_time = None, None

    def key_callback(key):
        nonlocal paused, single_step
        if key == 32:
            paused = not paused
        elif key in (ord("N"), ord("n")):
            single_step = True

    def advance():
        nonlocal next_log, next_sample
        row = controller.step()
        peaks["force_N"] = max(peaks["force_N"], row["contact_force_N"])
        peaks["penetration_mm"] = max(peaks["penetration_mm"], row["penetration_mm"])
        peaks["joint_speed_rad_s"] = max(peaks["joint_speed_rad_s"], float(np.linalg.norm(data.qvel)))
        if data.time >= next_sample:
            samples.append(row)
            next_sample += 0.02
        if data.time >= next_log:
            print(f"t={row['time_s']:5.1f}s {row['phase']:12s} "
                  f"F={row['force_N']:5.1f}N  轴向间隙={-row['axial_offset_mm']:6.2f}mm "
                  f"横向误差={row['lateral_error_mm']:5.2f}mm "
                  f"角误差={row['angle_error_deg']:5.2f}deg 接触数={row['contacts']} "
                  f"合爪={row['paw_active']} 单爪指令={row['paw_torque_Nm']:.3f} N·m", flush=True)
            next_log += 1.0

    try:
        if args.headless:
            while data.time < args.duration - 0.5 * model.opt.timestep:
                advance()
        else:
            from mujoco import viewer as mj_viewer
            print("点击视窗后：空格运行/暂停，N 单步。")
            with mj_viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
                viewer.cam.lookat[:] = [-0.1, 0.45, 0.3]
                viewer.cam.distance = 1.3
                viewer.cam.azimuth = 180
                viewer.cam.elevation = -20
                while viewer.is_running() and data.time < args.duration:
                    frame_start = time.monotonic()
                    if not paused or single_step:
                        # 每个渲染帧执行约 16 ms 物理步；仿真时间不依赖渲染/绘图速度。
                        count = 1 if single_step else max(1, round(1 / (60 * model.opt.timestep)))
                        single_step = False
                        for _ in range(count):
                            if data.time >= args.duration:
                                break
                            advance()
                    viewer.sync()
                    time.sleep(max(0, 1 / 60 - (time.monotonic() - frame_start)))
    except (RuntimeError, FloatingPointError) as error:
        failure = str(error)
        failure_time = float(data.time)
        peaks["force_N"] = max(peaks["force_N"], controller.last_contact_force_N)
        peaks["penetration_mm"] = max(peaks["penetration_mm"], controller.last_penetration_mm)
    finally:
        data.qfrc_applied[controller.dofs] = 0.0
    warnings = {mujoco.mjtWarning(i).name: int(w.number)
                for i, w in enumerate(data.warning) if w.number}
    healthy = failure is None and not warnings
    result = dict(completed=controller.state == "HOLD" and healthy,
                  stable_contact=controller.stable_contact_time >= 1.0 and healthy,
                  status="error" if not healthy else controller.state,
                  simulated_time_s=float(data.time), wall_time_s=time.monotonic() - started,
                  final=controller.metrics, peaks=peaks, warnings=warnings,
                  saturated_steps=controller.saturated_steps, failure=failure,
                  failure_time_s=failure_time)
    if healthy and controller.state == "CONTACT_HOLD":
        print("已稳定保压，但未完全贴合；保留现有碰撞模型，未强行推进到名义零间隙。")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.csv and samples:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.headless and not args.no_plot and samples:
        import matplotlib.pyplot as plt
        t = [r["time_s"] for r in samples]
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
        axes[0].plot(t, [r["force_N"] for r in samples], label="Interface force")
        axes[0].plot(t, [r["force_target_N"] for r in samples], "--", label="Target")
        axes[0].set_ylabel("Force (N)")
        axes[0].legend()
        axes[1].plot(t, [r["plane_residual_mm"] for r in samples])
        axes[1].set_ylabel("Mating plane residual (mm)")
        axes[1].set_xlabel("Simulation time (s)")
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        plt.show()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true", help="无界面验证模式")
    parser.add_argument("--autostart", action="store_true", help="打开窗口后自动运行")
    parser.add_argument("--no-plot", action="store_true", help="结束后不绘图")
    parser.add_argument("--duration", type=float, default=40.0, help="仿真时长（秒，默认 40）")
    parser.add_argument("--force", type=float, default=30.0, help="预压力 N，范围 5~60")
    parser.add_argument("--approach-speed", type=float, default=0.015, help="接近速度 m/s，范围 0.001~0.03")
    parser.add_argument("--csv", type=Path, help="保存 50 Hz 诊断数据")
    parser.add_argument("--summary", type=Path, help="保存 JSON 验证结果")
    parser.add_argument("--paw-gap-mm", type=float, default=6.0, help="合爪轴向距离阈值（mm）")
    add_arguments(parser)
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration 必须是有限正数")
    if not np.isfinite(args.paw_gap_mm) or args.paw_gap_mm <= 0:
        parser.error("--paw-gap-mm 必须是有限正数")
    if not 5 <= args.force <= 60:
        parser.error("force 必须在 5~60 N 内")
    if not 0.001 <= args.approach_speed <= 0.03:
        parser.error("approach-speed 必须在 0.001~0.03 m/s 内")
    try:
        result = run(args)
    except ValueError as error:
        parser.error(str(error))
    if result["failure"] or result["warnings"]:
        raise SystemExit(1)
    if args.headless and not result["completed"]:
        raise SystemExit(2)  # 到时仍未贴合，不把“程序未报错”等同于成功


if __name__ == "__main__":
    main()
