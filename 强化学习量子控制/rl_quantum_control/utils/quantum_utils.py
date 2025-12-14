"""
量子计算工具函数
===============
"""

import numpy as np
from typing import List, Tuple, Optional


def state_to_bloch(state: np.ndarray) -> Tuple[float, float, float]:
    """
    将单量子比特态转换为布洛赫球坐标
    
    Args:
        state: 量子态向量 [alpha, beta]
        
    Returns:
        (x, y, z) 布洛赫球坐标
    """
    if len(state) != 2:
        raise ValueError("只支持单量子比特态")
    
    alpha, beta = state
    
    # 计算布洛赫球坐标
    x = 2 * np.real(np.conj(alpha) * beta)
    y = 2 * np.imag(np.conj(alpha) * beta)
    z = np.abs(alpha)**2 - np.abs(beta)**2
    
    return x, y, z


def bloch_to_state(x: float, y: float, z: float) -> np.ndarray:
    """
    将布洛赫球坐标转换为量子态
    
    Args:
        x, y, z: 布洛赫球坐标
        
    Returns:
        量子态向量
    """
    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    
    return np.array([alpha, beta], dtype=np.complex128)


def random_unitary(dim: int) -> np.ndarray:
    """
    生成随机酉矩阵（Haar随机）
    
    Args:
        dim: 矩阵维度
        
    Returns:
        随机酉矩阵
    """
    # 使用QR分解生成Haar随机酉矩阵
    z = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    return q * ph


def random_state(n_qubits: int) -> np.ndarray:
    """
    生成随机量子态
    
    Args:
        n_qubits: 量子比特数
        
    Returns:
        随机量子态向量
    """
    dim = 2 ** n_qubits
    state = np.random.randn(dim) + 1j * np.random.randn(dim)
    return state / np.linalg.norm(state)


def state_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """
    计算两个量子态的保真度
    
    Args:
        state1, state2: 量子态向量
        
    Returns:
        保真度
    """
    state1 = state1 / np.linalg.norm(state1)
    state2 = state2 / np.linalg.norm(state2)
    return np.abs(np.vdot(state1, state2)) ** 2


def gate_fidelity(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    计算两个量子门的平均门保真度
    
    Args:
        U1, U2: 酉矩阵
        
    Returns:
        平均门保真度
    """
    d = U1.shape[0]
    trace = np.abs(np.trace(U1.conj().T @ U2))
    return (trace ** 2 + d) / (d * (d + 1))


def process_fidelity(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    计算过程保真度
    
    Args:
        U1, U2: 酉矩阵
        
    Returns:
        过程保真度
    """
    d = U1.shape[0]
    return np.abs(np.trace(U1.conj().T @ U2)) ** 2 / d ** 2


def diamond_distance_bound(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    钻石距离的上界估计
    
    Args:
        U1, U2: 酉矩阵
        
    Returns:
        钻石距离上界
    """
    d = U1.shape[0]
    f = process_fidelity(U1, U2)
    return np.sqrt(1 - f)


def pauli_decomposition(U: np.ndarray) -> dict:
    """
    将2x2酉矩阵分解为Pauli基
    
    Args:
        U: 2x2酉矩阵
        
    Returns:
        Pauli系数字典
    """
    if U.shape != (2, 2):
        raise ValueError("只支持2x2矩阵")
    
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    
    paulis = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    coeffs = {}
    
    for name, P in paulis.items():
        coeffs[name] = np.trace(P @ U) / 2
    
    return coeffs


def su2_decomposition(U: np.ndarray) -> Tuple[float, float, float, float]:
    """
    将SU(2)矩阵分解为ZYZ欧拉角
    
    Args:
        U: 2x2 SU(2)矩阵
        
    Returns:
        (global_phase, theta, phi, lambda) 参数
    """
    # 确保是SU(2)
    det = np.linalg.det(U)
    global_phase = np.angle(det) / 2
    U = U * np.exp(-1j * global_phase)
    
    # 提取参数
    theta = 2 * np.arccos(np.clip(np.abs(U[0, 0]), 0, 1))
    
    if np.abs(np.sin(theta / 2)) < 1e-10:
        phi = 0
        lam = 2 * np.angle(U[0, 0])
    elif np.abs(np.cos(theta / 2)) < 1e-10:
        phi = 2 * np.angle(U[1, 0])
        lam = 0
    else:
        phi = np.angle(U[1, 0]) - np.angle(U[0, 0])
        lam = np.angle(-U[0, 1]) - np.angle(U[0, 0])
    
    return global_phase, theta, phi, lam


def tensor_product(operators: List[np.ndarray]) -> np.ndarray:
    """
    计算算子的张量积
    
    Args:
        operators: 算子列表
        
    Returns:
        张量积结果
    """
    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)
    return result


def partial_trace(rho: np.ndarray, keep: List[int], 
                  dims: List[int]) -> np.ndarray:
    """
    计算偏迹
    
    Args:
        rho: 密度矩阵
        keep: 保留的子系统索引
        dims: 各子系统维度
        
    Returns:
        部分迹后的密度矩阵
    """
    n_systems = len(dims)
    keep = sorted(keep)
    trace_out = [i for i in range(n_systems) if i not in keep]
    
    # 重塑为张量
    rho_tensor = rho.reshape(dims + dims)
    
    # 迹掉不需要的子系统
    for idx in reversed(trace_out):
        rho_tensor = np.trace(rho_tensor, axis1=idx, axis2=idx + n_systems)
        n_systems -= 1
    
    # 重塑回矩阵
    new_dim = np.prod([dims[i] for i in keep])
    return rho_tensor.reshape(new_dim, new_dim)


def entanglement_entropy(state: np.ndarray, partition: int) -> float:
    """
    计算双分纠缠熵
    
    Args:
        state: 量子态向量
        partition: 分割位置（前partition个量子比特为A）
        
    Returns:
        纠缠熵
    """
    n_qubits = int(np.log2(len(state)))
    dims = [2] * n_qubits
    
    # 计算密度矩阵
    rho = np.outer(state, np.conj(state))
    
    # 计算约化密度矩阵
    keep = list(range(partition))
    rho_A = partial_trace(rho, keep, dims)
    
    # 计算冯诺依曼熵
    eigenvalues = np.linalg.eigvalsh(rho_A)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
    
    return entropy


def concurrence(rho: np.ndarray) -> float:
    """
    计算两量子比特混态的并发度
    
    Args:
        rho: 4x4密度矩阵
        
    Returns:
        并发度
    """
    if rho.shape != (4, 4):
        raise ValueError("需要4x4密度矩阵")
    
    # sigma_y ⊗ sigma_y
    sigma_y = np.array([[0, -1j], [1j, 0]])
    Y = np.kron(sigma_y, sigma_y)
    
    # R矩阵
    rho_tilde = Y @ rho.conj() @ Y
    R = rho @ rho_tilde
    
    # 特征值
    eigenvalues = np.sqrt(np.abs(np.linalg.eigvalsh(R)))
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    return max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])


if __name__ == "__main__":
    print("=== 量子工具函数测试 ===\n")
    
    # 测试布洛赫球转换
    print("1. 布洛赫球转换")
    state = np.array([1, 0], dtype=np.complex128)  # |0⟩
    x, y, z = state_to_bloch(state)
    print(f"|0⟩态布洛赫坐标: ({x:.3f}, {y:.3f}, {z:.3f})")
    
    state_plus = np.array([1, 1]) / np.sqrt(2)  # |+⟩
    x, y, z = state_to_bloch(state_plus)
    print(f"|+⟩态布洛赫坐标: ({x:.3f}, {y:.3f}, {z:.3f})")
    
    # 测试保真度
    print("\n2. 保真度计算")
    state1 = np.array([1, 0])
    state2 = np.array([1, 1]) / np.sqrt(2)
    f = state_fidelity(state1, state2)
    print(f"|0⟩和|+⟩的保真度: {f:.4f}")
    
    # 测试随机酉矩阵
    print("\n3. 随机酉矩阵")
    U = random_unitary(2)
    print(f"U†U ≈ I: {np.allclose(U.conj().T @ U, np.eye(2))}")
    
    # 测试纠缠熵
    print("\n4. 纠缠熵")
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)  # Bell态
    S = entanglement_entropy(bell, 1)
    print(f"Bell态纠缠熵: {S:.4f}")
    
    product = np.array([1, 0, 0, 0])  # |00⟩
    S_prod = entanglement_entropy(product, 1)
    print(f"|00⟩纠缠熵: {S_prod:.4f}")
