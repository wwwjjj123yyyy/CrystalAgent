import logging
from typing import Tuple, List, Dict, Any

# 引入或兼容 Pymatgen 核心库
try:
    from pymatgen.core import Structure
except ImportError:
    class Structure:
        def __init__(self, lattice, species, coords):
            self.lattice = lattice
            self.species = species
            self.frac_coords = coords
            self.distance_matrix = [[2.0 for _ in species] for _ in species]
            self.volume = 100.0
        @property
        def num_sites(self) -> int:
            return len(self.species)

# 按照架构设计，引入核心 Candidate 数据结构
try:
    from core.candidate import CrystalCandidate
except ImportError:
    # 兼容独立运行调试
    from typing import TypeVar
    CrystalCandidate = TypeVar('CrystalCandidate')

# 配置日志
logger = logging.getLogger("PhysicsValidator")
logger.setLevel(logging.INFO)


class CrystalValidator:
    """
    晶体结构物理硬约束校验器 (Physics Validator)
    在不调用高昂的 DFT/ML 势能评估前，通过纯几何与拓扑判据，
    快速熔断过滤掉掉在物理上“不可能存在”的爆炸或坍塌晶体。
    """
    def __init__(
        self, 
        min_bond_distance: float = 1.2, 
        min_vol_per_atom: float = 8.0, 
        max_vol_per_atom: float = 60.0
    ):
        """
        Args:
            min_bond_distance (float): 允许的最小键长（单位：Å）。小于此值视为原子空间内爆冲突。
            min_vol_per_atom (float): 每个原子平均占有的最小体积（单位：Å³）。防止过度挤压。
            max_vol_per_atom (float): 每个原子平均占有的最大体积（单位：Å³）。防止晶胞过度膨胀变为空旷真空。
        """
        self.min_bond_distance = min_bond_distance
        self.min_vol_per_atom = min_vol_per_atom
        self.max_vol_per_atom = max_vol_per_atom

    def check_atomic_overlap(self, structure: Structure) -> Tuple[bool, str]:
        """
        1. 检查原子重叠 (Atomic Overlap Check)
        利用距离矩阵，排查是否存在因突变（如填隙或坐标扰动）导致挨得太近的原子对。
        """
        if not hasattr(structure, "distance_matrix"):
            return True, "Passed (No distance matrix available)"

        dm = structure.distance_matrix
        num_atoms = len(dm)
        
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                dist = dm[i][j]
                if dist < self.min_bond_distance:
                    return False, f"Atomic clash detected: Atoms {i}({structure.species[i]}) and {j}({structure.species[j]}) are too close ({dist:.2f} Å < {self.min_bond_distance} Å)."
        
        return True, "Passed"

    def check_lattice_sanity(self, structure: Structure) -> Tuple[bool, str]:
        """
        2. 晶格合理性校验 (Lattice Sanity Check)
        检查晶胞体积与原子配比是否失衡，避免出现不切实际的宏观畸变。
        """
        num_atoms = structure.num_sites
        if num_atoms == 0:
            return False, "Lattice is empty (0 atoms)."

        # 计算每个原子平均分摊到的体积
        vol_per_atom = structure.volume / num_atoms

        if vol_per_atom < self.min_vol_per_atom:
            return False, f"Lattice implosion: Volume per atom is too small ({vol_per_atom:.2f} Å³ < {self.min_vol_per_atom} Å³)."
        
        if vol_per_atom > self.max_vol_per_atom:
            return False, f"Lattice explosion/vacuum: Volume per atom is too large ({vol_per_atom:.2f} Å³ > {self.max_vol_per_atom} Å³)."

        # 检查晶格夹角（避免晶格扭曲成极度扁平的平行六面体导致计算发散）
        if hasattr(structure, "lattice") and hasattr(structure.lattice, "angles"):
            for angle in structure.lattice.angles:
                if angle < 20.0 or angle > 160.0:
                    return False, f"Unstable lattice angle detected: {angle:.1f}° (Must be between 20° and 160°)."

        return True, "Passed"

    def validate(self, candidate: CrystalCandidate) -> Tuple[bool, str]:
        """
        主校验入口：串联所有物理硬约束判据。
        
        Returns:
            Tuple[bool, str]: (是否合法, 错误/成功报告)
        """
        struct = candidate.structure
        
        # 1. 基础结构存在性检查
        if struct is None:
            return False, "Structure object is None."

        # 2. 运行原子重叠冲突排查
        is_dist_ok, dist_msg = self.check_atomic_overlap(struct)
        if not is_dist_ok:
            return False, dist_msg

        # 3. 运行晶格宏观合理性排查
        is_lattice_ok, lattice_msg = self.check_lattice_sanity(struct)
        if not is_lattice_ok:
            return False, lattice_msg

        return True, "All physical constraints passed successfully."


# ==========================================
# 单元测试与管道集成演示
# ==========================================
if __name__ == "__main__":
    print("--- Testing CrystalValidator Component ---")
    
    # 模拟一个合法的晶体
    good_struct = Structure(None, ["Bi", "Te"], [[0,0,0], [0.5,0.5,0.5]])
    good_struct.volume = 40.0 # 20 Å³/atom, 在 8~60 范围内
    good_struct.distance_matrix = [[0.0, 2.5], [2.5, 0.0]] # 键长 2.5 Å
    
    # 模拟一个内爆冲突的非法晶体（突变失败产物）
    bad_struct = Structure(None, ["Bi", "Bi"], [[0,0,0], [0.01,0.01,0.01]])
    bad_struct.volume = 30.0
    bad_struct.distance_matrix = [[0.0, 0.5], [0.5, 0.0]] # 键长仅 0.5 Å，严重重叠
    
    # 转换为 Candidate
    class MockCandidate:
        def __init__(self, structure):
            self.id = "cand_test"
            self.structure = structure
            self.is_valid = True
            self.error_msg = ""

    cand_good = MockCandidate(good_struct)
    cand_bad = MockCandidate(bad_struct)

    # 实例化安检员
    validator = CrystalValidator()

    # 验证合法个体
    ok, msg = validator.validate(cand_good)
    print(f"Good Candidate Validation Result: Status={ok}, Message={msg}")

    # 验证非法个体
    ok, msg = validator.validate(cand_bad)
    print(f"Bad Candidate Validation Result: Status={ok}, Message={msg}")
