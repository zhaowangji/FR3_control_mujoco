"""在对齐后的 keti 对接面坐标系内定义初始偏差。"""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def add_arguments(parser):
    parser.add_argument("--xy-mm", nargs=2, type=float, default=(0.0, 0.0),
                        metavar=("X", "Y"), help="主动端中心沿对齐坐标系 X、Y 的偏移，单位 mm")
    parser.add_argument("--rpy-deg", nargs=3, type=float, default=(0.0, 0.0, 0.0),
                        metavar=("RX", "RY", "RZ"),
                        help="绕对齐坐标系固定 X、Y、Z 轴依次旋转，单位度")


def initialize_pose(model, data, body, qadr, dofs, aligned_R, seat, xy_mm, rpy_deg,
                    lower, upper, actuators=None):
    """先求仅有轴向间距的对齐解，再求指定偏差的解。仿真从第二个解开始。"""
    if not np.all(np.isfinite([*xy_mm, *rpy_deg])):
        raise ValueError("位置和姿态偏差必须是有限数")
    axis = aligned_R[:, 2]
    initial_axial = float(axis @ (data.xpos[body] - seat))
    aligned_pos = seat + axis * initial_axial
    shifted_pos = aligned_pos + aligned_R @ np.r_[np.asarray(xy_mm) / 1000.0, 0.0]
    shifted_R = aligned_R @ Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    radius = 0.15

    for label, target_pos, target_R in (
        ("对齐", aligned_pos, aligned_R), ("偏置", shifted_pos, shifted_R)
    ):
        for _ in range(150):
            mujoco.mj_kinematics(model, scratch)
            mujoco.mj_comPos(model, scratch)
            pos = scratch.xpos[body]
            rot = scratch.xmat[body].reshape(3, 3)
            dp = target_pos - pos
            dr = Rotation.from_matrix(target_R @ rot.T).as_rotvec()
            if np.linalg.norm(dp) < 1e-7 and np.linalg.norm(dr) < 1e-6:
                break
            mujoco.mj_jac(model, scratch, jacp, jacr, pos, body)
            jac = np.vstack((jacp[:, dofs], radius * jacr[:, dofs]))
            delta = np.linalg.lstsq(jac, np.r_[dp, radius * dr], rcond=1e-5)[0]
            delta *= min(1.0, 0.08 / max(np.max(np.abs(delta)), 1e-12))
            scratch.qpos[qadr] = np.clip(scratch.qpos[qadr] + delta, lower, upper)
        else:
            raise ValueError(f"{label}位姿的逆运动学未收敛，请检查偏差和关节行程")
        mujoco.mj_fwdPosition(model, scratch)
        if scratch.warning[mujoco.mjtWarning.mjWARN_CONTACTFULL].number:
            raise ValueError(f"{label}位姿接触数量过多")
        if any(con.dist < -1e-5 for con in scratch.contact):
            raise ValueError(f"{label}位姿已有碰撞，请减小偏差")

    data.qpos[qadr] = scratch.qpos[qadr]
    data.qvel[dofs] = 0.0
    if actuators is not None:
        data.ctrl[actuators] = data.qpos[qadr]
    mujoco.mj_forward(model, data)
    actual_xyz = 1000 * aligned_R.T @ (data.xpos[body] - aligned_pos)
    actual_rpy = Rotation.from_matrix(
        aligned_R.T @ data.xmat[body].reshape(3, 3)).as_euler("xyz", degrees=True)
    print(f"t=0 初始偏差实测 XYZ={actual_xyz.round(5)} mm，"
          f"RX/RY/RZ={actual_rpy.round(5)}°，名义轴向间距={-1000 * initial_axial:.3f} mm")
    return shifted_pos, shifted_R, initial_axial


class CaptureAssist:
    """按实测位姿渐进合爪。"""

    def __init__(self, model, gap, torque):
        self.paws = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator_paw{i}")
                     for i in (1, 2, 3)]
        if min(self.paws) < 0:
            raise ValueError("模型缺少夹爪执行器 actuator_paw1～3")
        self.gap = gap
        self.torque = torque
        self.active = False
        self.grip = 0.0

    def update_paws(self, data, dt, gap, lateral, angle):
        if self.active:
            if abs(gap) > self.gap + 0.002 or lateral > 0.003 or angle > np.deg2rad(2.5):
                self.active = False
        elif (abs(gap) <= self.gap
              and lateral <= 0.002 and angle <= np.deg2rad(1.5)):
            self.active = True
        self.grip = float(np.clip(self.grip + (1 if self.active else -1) * dt / 1.25, 0.0, 1.0))
        data.ctrl[self.paws] = self.torque * self.grip**2 * (3.0 - 2.0 * self.grip)
