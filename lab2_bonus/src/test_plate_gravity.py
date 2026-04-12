import numpy as np
import bonus_plate_gravity as bpg

# 测试z在[0.2, 10]范围内的结果
def test_force_curve():
    # 生成z值数组
    z_values = np.linspace(0.2, 10, 50)
    
    # 计算力曲线
    Fz_values = bpg.force_curve(z_values)
    
    # 打印结果
    print("z (m)\tFz (N)")
    print("------------------------")
    for z, Fz in zip(z_values, Fz_values):
        print(f"{z:.2f}\t{Fz:.10e}")

if __name__ == "__main__":
    test_force_curve()
