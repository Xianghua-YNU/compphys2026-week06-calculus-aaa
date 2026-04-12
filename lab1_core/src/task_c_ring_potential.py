import numpy as np
import matplotlib.pyplot as plt


def ring_potential_point(x: float, y: float, z: float, a: float = 1.0, q: float = 1.0, n_phi: int = 720) -> float:
    """
    用离散梯形积分计算空间任意点(x,y,z)处的均匀带电圆环电势
    :param x, y, z: 空间点坐标
    :param a: 圆环半径，默认1.0
    :param q: 归一化总电荷，默认1.0
    :param n_phi: 角度φ的采样点数，默认720（高精度）
    :return: 电势V
    """
    phi = np.linspace(0, 2 * np.pi, n_phi)
    r = np.sqrt(
        (x - a * np.cos(phi))**2 +
        (y - a * np.sin(phi))**2 +
        z**2
    )
    r[r < 1e-10] = 1e-10  # 避免除零
    # 适配新版NumPy，使用标准梯形积分函数
    integral = np.trapezoid(1 / r, phi)
    return (q / (2 * np.pi)) * integral


def ring_potential_grid(y_grid: np.ndarray, z_grid: np.ndarray, x0: float = 0.0,
                        a: float = 1.0, q: float = 1.0, n_phi: int = 720) -> np.ndarray:
    """
    计算x=x0平面（默认yz平面）上的网格电势分布
    :param y_grid: y方向坐标数组（一维，或meshgrid生成的二维数组）
    :param z_grid: z方向坐标数组（一维，或meshgrid生成的二维数组）
    :param x0: 平面x坐标，默认0.0（yz平面）
    :param a: 圆环半径，默认1.0
    :param q: 归一化总电荷，默认1.0
    :param n_phi: 角度φ的采样点数，默认720
    :return: 电势网格V_grid，shape为 (len(z_grid), len(y_grid))（输入为一维时）
             或与输入二维数组一致（输入为meshgrid后时）
    """
    # 1. 统一输入格式：自动处理一维/二维输入
    if y_grid.ndim == 1 and z_grid.ndim == 1:
        # 输入为一维坐标数组，生成meshgrid二维网格
        Y, Z = np.meshgrid(y_grid, z_grid, indexing='xy')
    elif y_grid.ndim == 2 and z_grid.ndim == 2:
        # 输入为已生成的二维meshgrid数组，直接使用
        Y, Z = y_grid, z_grid
    else:
        raise ValueError("y_grid和z_grid必须同时为一维数组，或同时为二维meshgrid数组")
    
    # 2. 初始化电势网格，shape与二维Y完全一致
    V_grid = np.zeros_like(Y, dtype=np.float64)
    
    # 3. 遍历网格点计算电势（无索引越界风险）
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            V_grid[i, j] = ring_potential_point(
                x=x0, y=Y[i, j], z=Z[i, j],
                a=a, q=q, n_phi=n_phi
            )
    
    return V_grid


def axis_potential_analytic(z: float, a: float = 1.0, q: float = 1.0) -> float:
    """
    均匀带电圆环对称轴（x=0,y=0）上的电势解析解
    :param z: 轴上点的z坐标
    :param a: 圆环半径，默认1.0
    :param q: 归一化总电荷，默认1.0
    :return: 轴上电势V
    """
    return q / np.sqrt(a**2 + z**2)


# ==================== 主程序：测试验证 ====================
if __name__ == "__main__":
    # 复现测试用例，验证修复效果
    ys = np.linspace(-0.5, 0.5, 11)
    zs = np.linspace(-0.5, 0.5, 13)
    try:
        V = ring_potential_grid(ys, zs, x0=0.0, a=1.0, q=1.0, n_phi=360)
        print(f"测试通过！电势网格shape: {V.shape}（预期(13,11)，实际{V.shape}）")
    except Exception as e:
        print(f"测试失败: {str(e)}")

    # ==================== 绘图：带电圆环电势场（Task C 必须） ====================
    # 1. 生成网格
    y = np.linspace(-2.0, 2.0, 100)
    z = np.linspace(-2.0, 2.0, 100)  # 已修复：lspace → linspace
    Y, Z = np.meshgrid(y, z)

    # 2. 计算电势网格
    V = ring_potential_grid(Y, Z, x0=0.0, a=1.0, q=1.0)

    # 3. 绘制等势线图
    plt.figure(figsize=(7, 6))
    cp = plt.contour(Y, Z, V, levels=30, cmap='coolwarm')
    plt.clabel(cp, inline=True, fontsize=8)
    plt.title('带电圆环在 yz 平面的等势线分布')
    plt.xlabel('y')
    plt.ylabel('z')
    plt.axis('equal')
    plt.grid(alpha=0.3)
    plt.savefig('ring-potential.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. 验证：轴上解析解 vs 数值解
    z_test = np.linspace(-2, 2, 50)
    V_num = [ring_potential_point(0, 0, z) for z in z_test]
    V_ana = axis_potential_analytic(z_test)

    plt.figure(figsize=(7,3))
    plt.plot(z_test, V_ana, 'r-', label='解析解')
    plt.plot(z_test, V_num, 'bo', markersize=2, label='数值解')
    plt.xlabel('z')
    plt.ylabel('电势 V')
    plt.title('z轴电势：数值解 vs 解析解')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()