import ikpy.chain
import ikpy.utils.plot as plot_utils
import matplotlib.pyplot as plt
import transforms3d as tf
 
def main():
    my_chain = ikpy.chain.Chain.from_urdf_file("model/ur5e.urdf")
    ee_pos = [-0.1, 0.3, 0.1] # 末端执行器在世界坐标系下的目标位置的xyz 
    ee_euler = [3.14, 0, 1.57] # 末端执行器的目标欧拉角姿态（弧度）
    ee_orientation = tf.euler.euler2mat(*ee_euler) # 将欧拉角姿态用旋转矩阵表示
    ref_pos = [0, 0, -1.57, -1.34, 2.65, -1.3, 1.55, 0, 0] # 提供初始关节角猜测，用于优化求解
    
    fig, ax = plot_utils.init_3d_figure()
    my_chain.plot(my_chain.inverse_kinematics(ee_pos, ee_orientation, "all", initial_position=ref_pos), ax)
    plt.show()
 
if __name__ == "__main__":
    main()