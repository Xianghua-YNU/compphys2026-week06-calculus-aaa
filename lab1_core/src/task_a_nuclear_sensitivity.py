import numpy as np

def rate_3alpha(T: float) -> float:
    """
    严格按照题目公式计算3α核反应率
    q(T) = 5.09×10¹¹ × T₈⁻³ × exp(-44.027 / T₈)，其中 T₈ = T / 10⁸
    :param T: 温度 (单位：K)，必须大于0
    :return: 3α核反应率
    """
    if T <= 0:
        raise ValueError("温度T必须大于0")
    T8 = T / 1e8
    q = 5.09e11 * (T8 ** (-3)) * np.exp(-44.027 / T8)
    return q

def finite_diff_dq_dT(T0, h=1e-8):
    """
    用前向差分近似计算dq/dT，严格遵循题目要求：ΔT = h·T₀
    :param T0: 参考温度 (单位：K)
    :param h: 相对步长，默认1e-8（平衡截断误差与舍入误差）
    :return: 导数 dq/dT
    """
    if T0 <= 0:
        raise ValueError("温度T0必须大于0")
    delta_T = h * T0  # 关键修正：相对增量，而非绝对h
    q0 = rate_3alpha(T0)
    q1 = rate_3alpha(T0 + delta_T)
    dq_dT = (q1 - q0) / delta_T
    return dq_dT

def sensitivity_nu(T0, h=1e-8):
    """
    计算温度敏感性指数 ν = (T/q)·(dq/dT)
    :param T0: 参考温度 (单位：K)
    :param h: 差分步长
    :return: 敏感性指数 ν
    """
    q0 = rate_3alpha(T0)
    if q0 <= 1e-300:  # 避免数值下溢导致除以0
        raise ValueError(f"温度{T0:.2e} K下反应率数值下溢，无法计算ν")
    dq_dT = finite_diff_dq_dT(T0, h)
    nu = (T0 / q0) * dq_dT
    return nu

def nu_table(T_values, h=1e-8):
    """
    生成温度-敏感性指数对照表，格式为[(T0, nu0), (T1, nu1), ...]
    :param T_values: 温度列表/数组
    :param h: 差分步长
    :return: 对照表列表
    """
    result = []
    for T in T_values:
        try:
            nu = sensitivity_nu(T, h)
            result.append((T, nu))
        except Exception as e:
            print(f"⚠️ 温度{T:.2e} K计算失败：{str(e)}")
    return result

# ------------------- 测试与输出 -------------------
if __name__ == "__main__":
    # 题目要求的必算温度点
    T_required = [1.0e8, 2.5e8, 5.0e8, 1.0e9, 2.5e9, 5.0e9]
    h_default = 1e-8  # 题目建议的默认步长
    
    # 生成对照表
    table = nu_table(T_required, h_default)
    
    # 按题目要求的格式输出
    print("="*60)
    print("3α反应温度敏感性指数计算结果")
    print("="*60)
    for T, nu in table:
        print(f"{T:.3e} K : nu = {nu:.2f}")