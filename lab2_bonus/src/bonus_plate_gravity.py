import numpy as np


G = 6.674e-11


def gauss_legendre_2d(func, ax: float, bx: float, ay: float, by: float, n: int = 40) -> float:
    # 一维高斯-勒让德积分的节点和权重
    x, w = np.polynomial.legendre.leggauss(n)
    
    # 映射到积分区间 [ax, bx] 和 [ay, by]
    x_mapped = 0.5 * (bx - ax) * x + 0.5 * (bx + ax)
    y_mapped = 0.5 * (by - ay) * x + 0.5 * (by + ay)
    
    # 计算积分
    integral = 0.0
    for i in range(n):
        for j in range(n):
            integral += w[i] * w[j] * func(x_mapped[i], y_mapped[j])
    
    # 乘以积分区间的缩放因子
    integral *= 0.25 * (bx - ax) * (by - ay)
    
    return integral


def plate_force_z(z: float, L: float = 10.0, M_plate: float = 1.0e4, m_particle: float = 1.0, n: int = 40) -> float:
    # 计算面密度
    sigma = M_plate / (L ** 2)
    
    # 定义积分函数
    def integrand(x, y):
        r = np.sqrt(x**2 + y**2 + z**2)
        return 1 / (r**3)
    
    # 积分区间：x从-L/2到L/2，y从-L/2到L/2
    ax = -L / 2
    bx = L / 2
    ay = -L / 2
    by = L / 2
    
    # 计算积分
    integral = gauss_legendre_2d(integrand, ax, bx, ay, by, n)
    
    # 计算Fz
    Fz = G * sigma * m_particle * z * integral
    
    return Fz


def force_curve(z_values, L: float = 10.0, M_plate: float = 1.0e4, m_particle: float = 1.0, n: int = 40):
    # 计算每个z值对应的Fz
    Fz_values = []
    for z in z_values:
        Fz = plate_force_z(z, L, M_plate, m_particle, n)
        Fz_values.append(Fz)
    
    return np.array(Fz_values)
