import math
from scipy.integrate import quad

# 德拜积分被积函数（处理x→0边界）
def debye_integrand(x: float) -> float:
    if x < 1e-6:
        return x**2 + (5/12) * x**4  # x→0泰勒展开
    ex = math.exp(x)
    return (x**4 * ex) / (ex - 1)**2

# 复合梯形积分法
def trapezoid_composite(f, a: float, b: float, n: int) -> float:
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += 2 * f(a + i * h)
    return total * h / 2

# 复合Simpson积分法（强制校验n为偶数）
def simpson_composite(f, a: float, b: float, n: int) -> float:
    if n % 2 != 0:
        raise ValueError("Simpson法要求分段数n为偶数")
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        if i % 2 == 1:
            total += 4 * f(a + i * h)
        else:
            total += 2 * f(a + i * h)
    return total * h / 3

# 德拜积分主函数
def debye_integral(T: float, theta_d: float, method: str, n: int) -> float:
    y = theta_d / T  # 积分上限 y = θ_D/T
    a, b = 0.0, y
    if method == "trapezoid":
        return trapezoid_composite(debye_integrand, a, b, n)
    elif method == "simpson":
        return simpson_composite(debye_integrand, a, b, n)
    else:
        raise ValueError("仅支持'trapezoid'/'simpson'两种方法")

# ==================== 主程序：自动计算并输出表格 ====================
if __name__ == "__main__":
    # 统一测试参数（可按需修改）
    THETA_D = 300.0  # 德拜温度(K)
    T = 300.0        # 系统温度(K)
    n = 100          # 分段数（满足Simpson偶数要求）

    # 1. 计算高精度精确参考值
    I_exact, _ = quad(debye_integrand, 0, THETA_D/T)

    # 2. 计算两种方法的积分值与绝对误差
    I_trap = debye_integral(T, THETA_D, "trapezoid", n)
    err_trap = abs(I_trap - I_exact)
    
    I_simp = debye_integral(T, THETA_D, "simpson", n)
    err_simp = abs(I_simp - I_exact)

    # 3. 格式化输出表格（完全匹配题目表头格式）
    print("-" * 120)
    print(f"{'方法':<12} {'n':<8} {'积分值':<18} {'误差估计':<18} {'结论':<30}")
    print("-" * 120)
    # 梯形法数据行
    print(f"{'梯形法':<12} {n:<8} {I_trap:<18.6f} {err_trap:<18.2e} {'二阶收敛，精度低，收敛慢':<30}")
    # Simpson法数据行
    print(f"{'Simpson法':<12} {n:<8} {I_simp:<18.6f} {err_simp:<18.2e} {'四阶收敛，精度高，收敛快':<30}")
    print("-" * 120)

    # 可选：打印精确参考值用于验证
    print(f"\n[参考] 精确积分值（scipy高精度计算）：{I_exact:.10f}")