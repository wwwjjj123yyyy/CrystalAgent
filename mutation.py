import copy
import random
import logging
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional

# 引入或兼容 Pymatgen 核心库
try:
    from pymatgen.core import Structure, Lattice
except ImportError:
    # 保证在没有安装 pymatgen 的沙箱环境中脚本依旧可预测运行
    class Structure:
        def __init__(self, lattice, species, coords):
            self.lattice = lattice
            self.species = species
            self.frac_coords = coords
            self.distance_matrix = [[2.0 for _ in species] for _ in species]
        def copy(self): return copy.deepcopy(self)
        def perturb(self, distance): pass
        def apply_strain(self, strain): pass
        def make_supercell(self, scaling): pass
        @property
        def formula(self): return "MockFormula"

try:
    from candidate import CrystalCandidate
except ImportError:
    class CrystalCandidate:
        def __init__(self, structure: Structure, generation: int, parent_ids: List[str] = None):
            self.id = f"cand_{random.randint(1000,9999)}"
            self.structure = structure
            self.generation = generation
            self.parent_ids = parent_ids or []
            self.is_valid = False
            self.error_msg = ""
            self.fitness = 0.0

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MutationSystem")

# 全局化学替换元素池
SPECIES_POOL: Dict[str, List[str]] = {
    "Bi": ["Sb", "As"], "Sb": ["Bi", "As"],
    "Te": ["Se", "S"],  "Se": ["Te", "S"],
    "Sn": ["Ge", "Pb"]
}


# ==========================================
# 1. 枚举类型 (Enum)
# ==========================================

class MutationStrength(Enum):
    """突变强度分级"""
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"


class MutationType(Enum):
    """10种核心变异算子类型"""
    REPLACE_SPECIES = "replace_species"
    PERTURB_ATOMS = "perturb_atoms"
    SCALE_LATTICE = "scale_lattice"
    SHEAR_LATTICE = "shear_lattice"
    SYMMETRY_BREAK = "symmetry_break"
    CREATE_VACANCY = "create_vacancy"
    ADD_INTERSTITIAL = "add_interstitial"
    LAYER_SHIFT = "layer_shift"
    COMPOSITION_MUTATION = "composition_mutation"
    SUPERCELL_MUTATION = "supercell_mutation"


# ==========================================
# 2. 突变底库 (MutationLibrary)
# ==========================================

class MutationLibrary:
    """底层晶体几何与拓扑变异方法库"""

    @staticmethod
    def replace_species(structure: Structure, strength: MutationStrength) -> Structure:
        """1. 同族/同价元素替换"""
        new_struct = structure.copy()
        ratio = {MutationStrength.SMALL: 0.1, MutationStrength.MEDIUM: 0.3, MutationStrength.LARGE: 0.5}[strength]
        avail = [i for i, sp in enumerate(new_struct.species) if str(sp) in SPECIES_POOL]
        if not avail: return new_struct
        
        to_mod = random.sample(avail, max(1, int(len(avail) * ratio)))
        if hasattr(new_struct, "replace_species") and not isinstance(new_struct.species, list):
            for idx in to_mod:
                new_struct.replace_species({idx: random.choice(SPECIES_POOL[str(new_struct.species[idx])])})
        else:
            new_sp = list(new_struct.species)
            for idx in to_mod:
                new_sp[idx] = random.choice(SPECIES_POOL[str(new_sp[idx])])
            new_struct.species = new_sp
        return new_struct

    @staticmethod
    def perturb_atoms(structure: Structure, strength: MutationStrength) -> Structure:
        """2. 原子位置微扰 (Small: 0.03Å, Medium: 0.08Å, Large: 0.15Å)"""
        new_struct = structure.copy()
        dist = {MutationStrength.SMALL: 0.03, MutationStrength.MEDIUM: 0.08, MutationStrength.LARGE: 0.15}[strength]
        if hasattr(new_struct, "perturb"):
            new_struct.perturb(dist)
        return new_struct

    @staticmethod
    def scale_lattice(structure: Structure, strength: MutationStrength) -> Structure:
        """3. 晶格各向同性缩放 (Small: 1%, Medium: 3%, Large: 8%)"""
        new_struct = structure.copy()
        scale = {MutationStrength.SMALL: 0.01, MutationStrength.MEDIUM: 0.03, MutationStrength.LARGE: 0.08}[strength]
        factor = 1.0 + random.choice([-1, 1]) * scale
        if hasattr(new_struct, "apply_strain"):
            new_struct.apply_strain(factor - 1.0)
        return new_struct

    @staticmethod
    def shear_lattice(structure: Structure, strength: MutationStrength) -> Structure:
        """4. 晶格剪切形变"""
        new_struct = structure.copy()
        shear = {MutationStrength.SMALL: 0.02, MutationStrength.MEDIUM: 0.05, MutationStrength.LARGE: 0.10}[strength]
        if hasattr(new_struct, "apply_strain"):
            new_struct.apply_strain([0, 0, 0, shear, 0, 0])
        return new_struct

    @staticmethod
    def symmetry_break(structure: Structure, strength: MutationStrength) -> Structure:
        """5. 非均匀畸变导致空间群对称性破缺"""
        new_struct = structure.copy()
        disp = {MutationStrength.SMALL: 0.01, MutationStrength.MEDIUM: 0.04, MutationStrength.LARGE: 0.10}[strength]
        if hasattr(new_struct, "translate_sites"):
            targets = random.sample(range(len(new_struct.species)), max(1, len(new_struct.species) // 3))
            for idx in targets:
                new_struct.translate_sites([idx], [random.uniform(-disp, disp) for _ in range(3)], frac_coords=True)
        return new_struct

    @staticmethod
    def create_vacancy(structure: Structure, strength: MutationStrength) -> Structure:
        """6. 点缺陷：创建原子空位"""
        new_struct = structure.copy()
        count = {MutationStrength.SMALL: 1, MutationStrength.MEDIUM: 2, MutationStrength.LARGE: 4}[strength]
        to_rem = sorted(random.sample(range(len(new_struct.species)), min(count, len(new_struct.species) - 1)), reverse=True)
        if hasattr(new_struct, "remove_sites"):
            new_struct.remove_sites(to_rem)
        return new_struct

    @staticmethod
    def add_interstitial(structure: Structure, strength: MutationStrength) -> Structure:
        """7. 点缺陷：添加填隙原子"""
        new_struct = structure.copy()
        count = {MutationStrength.SMALL: 1, MutationStrength.MEDIUM: 2, MutationStrength.LARGE: 3}[strength]
        elements = list(set([str(s) for s in new_struct.species])) + ["O"]
        for _ in range(count):
            if hasattr(new_struct, "append"):
                new_struct.append(random.choice(elements), [random.random() for _ in range(3)], coords_are_fractional=True)
        return new_struct

    @staticmethod
    def layer_shift(structure: Structure, strength: MutationStrength) -> Structure:
        """8. 层错/原子层滑移 (Small: 0.10Å, Medium: 0.30Å, Large: 0.60Å)"""
        new_struct = structure.copy()
        shift = {MutationStrength.SMALL: 0.10, MutationStrength.MEDIUM: 0.30, MutationStrength.LARGE: 0.60}[strength]
        if hasattr(new_struct, "translate_sites") and hasattr(new_struct, "frac_coords"):
            upper_layer = [i for i, c in enumerate(new_struct.frac_coords) if c[2] > 0.5]
            if upper_layer:
                new_struct.translate_sites(upper_layer, [shift / 5.0, 0.0, 0.0], frac_coords=True)
        return new_struct

    @staticmethod
    def composition_mutation(structure: Structure, strength: MutationStrength) -> Structure:
        """9. 改变局部化学计量比"""
        new_struct = structure.copy()
        all_sp = [str(s) for s in new_struct.species]
        if len(set(all_sp)) >= 2:
            src, dst = random.sample(list(set(all_sp)), 2)
            idx_list = [i for i, s in enumerate(all_sp) if s == src]
            num = {MutationStrength.SMALL: 1, MutationStrength.MEDIUM: 2, MutationStrength.LARGE: 3}[strength]
            if idx_list and hasattr(new_struct, "replace_species"):
                for idx in idx_list[:min(num, len(idx_list))]:
                    new_struct.replace_species({idx: dst})
        return new_struct

    @staticmethod
    def supercell_mutation(structure: Structure, strength: MutationStrength) -> Structure:
        """10. 扩胞变异"""
        new_struct = structure.copy()
        matrix = {
            MutationStrength.SMALL: [2, 1, 1],
            MutationStrength.MEDIUM: [2, 2, 1],
            MutationStrength.LARGE: [2, 2, 2]
        }[strength]
        if hasattr(new_struct, "make_supercell"):
            try: new_struct.make_supercell(matrix)
            except Exception: pass
        return new_struct


# ==========================================
# 3. 统计看板组件 (MutationStatistics)
# ==========================================

class MutationStatistics:
    """独立统计模块：维护算子的成功率、失败率和历史平均收益（Fitness）"""
    def __init__(self):
        self.stats_data: Dict[str, Dict[str, Any]] = {
            m_type.value: {
                "success": 0,
                "failed": 0,
                "total_fitness": 0.0,
                "average_fitness": 0.0
            } for m_type in MutationType
        }

    def record(self, m_type: MutationType, is_success: bool, fitness: float = 0.0) -> None:
        """记录单次演化结果"""
        key = m_type.value
        data = self.stats_data[key]
        if is_success:
            data["success"] += 1
            data["total_fitness"] += fitness
            data["average_fitness"] = round(data["total_fitness"] / data["success"], 4)
        else:
            data["failed"] += 1

    def get_report(self) -> Dict[str, Dict[str, Any]]:
        """获取无隐藏因子的纯净数据报表，供 Planner (LLM) 决策参考"""
        return {
            k: {
                "success": v["success"],
                "failed": v["failed"],
                "average_fitness": v["average_fitness"]
            } for k, v in self.stats_data.items()
        }


# ==========================================
# 4. 突变控制调度核心 (MutationEngine)
# ==========================================

class MutationEngine:
    """突变调度核心：集成底层算子库，并在运行时调用 MutationStatistics 记录演化轨迹"""
    def __init__(self):
        # 绑定统计看板
        self.statistics = MutationStatistics()
        
        # 建立算子类型到具体执行函数的映射
        self.operator_map = {
            MutationType.REPLACE_SPECIES: MutationLibrary.replace_species,
            MutationType.PERTURB_ATOMS: MutationLibrary.perturb_atoms,
            MutationType.SCALE_LATTICE: MutationLibrary.scale_lattice,
            MutationType.SHEAR_LATTICE: MutationLibrary.shear_lattice,
            MutationType.SYMMETRY_BREAK: MutationLibrary.symmetry_break,
            MutationType.CREATE_VACANCY: MutationLibrary.create_vacancy,
            MutationType.ADD_INTERSTITIAL: MutationLibrary.add_interstitial,
            MutationType.LAYER_SHIFT: MutationLibrary.layer_shift,
            MutationType.COMPOSITION_MUTATION: MutationLibrary.composition_mutation,
            MutationType.SUPERCELL_MUTATION: MutationLibrary.supercell_mutation,
        }

    def apply(self, structure: Structure, m_type: MutationType, strength: MutationStrength) -> Structure:
        """应用指定的单步突变"""
        if m_type in self.operator_map:
            return self.operator_map[m_type](structure, strength)
        raise ValueError(f"Unknown mutation type: {m_type}")

    def random_apply(self, structure: Structure, strength: MutationStrength) -> Tuple[Structure, MutationType]:
        """随机挑选一种突变算子应用，并返回被挑中的算子类型"""
        chosen_type = random.choice(list(MutationType))
        return self.apply(structure, chosen_type, strength), chosen_type

    def apply_sequence(self, structure: Structure, sequence: List[Tuple[MutationType, MutationStrength]]) -> Structure:
        """串联执行一系列突变操作流水线"""
        res = structure.copy()
        for m_type, strength in sequence:
            res = self.apply(res, m_type, strength)
        return res


# ==========================================
# 5. 物理硬约束与完整工作流管道 (PhysicsConstraint)
# ==========================================

class PhysicsConstraint:
    """
    管道守门员：包含物理短键重叠排查、晶格推开修复（Repair）与最终合规校验（Validator）
    执行流程：Parent -> Mutation Engine -> Physics Constraint -> Structure Repair -> Validator -> Child
    """
    def __init__(self, min_bond_distance: float = 1.3):
        self.min_bond_distance = min_bond_distance

    def check_overlap(self, structure: Structure) -> List[Tuple[int, int, float]]:
        """原子碰撞挤压检测"""
        overlaps = []
        if hasattr(structure, "distance_matrix"):
            dm = structure.distance_matrix
            for i in range(len(dm)):
                for j in range(i + 1, len(dm)):
                    if dm[i][j] < self.min_bond_distance:
                        overlaps.append((i, j, dm[i][j]))
        return overlaps

    def repair_structure(self, structure: Structure, overlaps: List[Tuple[int, int, float]]) -> Structure:
        """结构修复层 (Structure Repair)：微调重叠原子的坐标以平滑晶格畸变"""
        repaired = structure.copy()
        if hasattr(repaired, "translate_sites"):
            for i, j, _ in overlaps:
                repaired.translate_sites([i], [0.02, 0.01, 0.0], frac_coords=True)
                repaired.translate_sites([j], [-0.02, -0.01, 0.0], frac_coords=True)
        return repaired

    def validator(self, structure: Structure) -> Tuple[bool, str]:
        """终审校验器 (Validator)"""
        if len(structure.species) < 2:
            return False, "Failed: Atomic structural collapse."
        if len(self.check_overlap(structure)) > 0:
            return False, "Failed: Severe irreversible atomic overlaps."
        return True, "Passed constraints"

    def execute_workflow(self, parent: CrystalCandidate, engine: MutationEngine,
                         m_type: Optional[MutationType] = None,
                         strength: MutationStrength = MutationStrength.MEDIUM,
                         generation: int = 1) -> Tuple[CrystalCandidate, MutationType]:
        """
        完整执行生命周期流水线，并在前期校验失败时直接触发统计数据回填
        """
        # 1. 突变阶段
        chosen_type = m_type if m_type else random.choice(list(MutationType))
        mutated_struct = engine.apply(parent.structure, chosen_type, strength)
        
        # 2 & 3. 物理冲突检测与自动修复阶段
        overlaps = self.check_overlap(mutated_struct)
        if overlaps:
            fixed_struct = self.repair_structure(mutated_struct, overlaps)
        else:
            fixed_struct = mutated_struct
            
        # 4. 最终评估合规审查 (Validator)
        is_legit, report_msg = self.validator(fixed_struct)
        
        # 5. 生成 Child Candidate
        child = CrystalCandidate(structure=fixed_struct, generation=generation, parent_ids=[parent.id])
        child.is_valid = is_legit
        child.error_msg = report_msg
        
        # 如果在 Validator 物理层就发生崩溃或不可修复，直接在此处标记 Failed
        if not is_legit:
            engine.statistics.record(chosen_type, is_success=False, fitness=0.0)
            
        return child, chosen_type


# ==========================================
# 6. 模拟闭环运行验证
# ==========================================
if __name__ == "__main__":
    # 创建模拟母体
    mock_structure = Structure(None, ["Bi", "Bi", "Te", "Te"], [[0,0,0], [0.5,0.5,0.1], [0.5,0,0.5], [0,0.5,0.5]])
    parent = CrystalCandidate(structure=mock_structure, generation=0)
    
    # 实例化引擎与物理管道
    engine = MutationEngine()
    pipeline = PhysicsConstraint()
    
    print("=== Running Evolution Pipeline Loops ===")
    
    # 模拟执行 3 次突变搜索尝试
    for i in range(3):
        child, op_used = pipeline.execute_workflow(parent, engine, strength=MutationStrength.MEDIUM, generation=1)
        
        if child.is_valid:
            # 模拟在流水线外部（主循环中）通过评估器得到了高适应度得分
            simulated_fitness = round(random.uniform(0.6, 0.95), 4)
            child.fitness = simulated_fitness
            
            # 将成功信息以及真实的 Fitness 回填至引擎的统计模块
            engine.statistics.record(op_used, is_success=True, fitness=simulated_fitness)
            print(f"Loop {i+1}: Mutation [{op_used.value}] Success! Fitness assigned: {simulated_fitness}")
        else:
            print(f"Loop {i+1}: Mutation [{op_used.value}] Failed. Reason: {child.error_msg}")

    # 导出统计报表给 Planner
    print("\n=== Final Mutation Statistics Report (For Planner Reflection) ===")
    report = engine.statistics.get_report()
    import json
    print(json.dumps(report, indent=4))
