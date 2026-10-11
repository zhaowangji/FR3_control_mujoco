import mujoco.viewer
import time
import ikpy.chain
import transforms3d as tf
 
def main():
    model = mujoco.MjModel.from_xml_path('model/universal_robots_ur5e/scene.xml')
    data = mujoco.MjData(model)
    my_chain = ikpy.chain.Chain.from_urdf_file("model/ur5e.urdf")
 
    ee_pos = [-0.1, 0.3, 0.1]
    ee_euler = [3.14, 0, 1.57]
    ref_pos = [0, 0, -1.57, -1.34, 2.65, -1.3, 1.55, 0, 0]
    ee_orientation = tf.euler.euler2mat(*ee_euler)
    ee_id = model.site("attachment_site").id # 获取末端执行器对应的id
 
    joint_angles = my_chain.inverse_kinematics(ee_pos, ee_orientation, "all", initial_position=ref_pos)
    ctrl = joint_angles[2:-1]
    data.ctrl[:6] = ctrl
 
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)
 
    ee_pos = data.site_xpos[ee_id] # 获取此时末端执行器对应的xyz坐标信息
    print(ee_pos)
    print(joint_angles[2:-1]) # 打印此时机械臂的关节角度
 
if __name__ == "__main__":
    main()