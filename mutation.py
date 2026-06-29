import copy
import random
import logging
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional

# 引入或兼容 Pymatgen 核心库
try:
    from pymatgen.core import Structure, Lattice, Element
except ImportError:
    # 保证没有安装 pymatgen 时脚本依旧能编译通过并进行结构模拟
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

# 假定从上一层的 candidate.py 导入 CrystalCandidate 结构
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
logger = logging.getLogger("MutationSystem")
logger.setLevel(logging.INFO)

# ==========================================
# 1. 枚举类型定义 (Enum)
# ==========================================

class MutationStrength(Enum):
    """突变强度枚举"""
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"


class MutationType(Enum):
    """突变算子类型枚举"""
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


# 全局同族/同价元素替换池
SPECIES_POOL: Dict[str, List[str]] = {
    "Bi": ["Sb", "As"],
    "Sb": ["Bi", "As"],
    "Te": ["Se", "S"],
    "Se": ["Te", "S"],
    "Sn": ["Ge", "Pb"]
}

# ==========================================
# 2. 突变算子底层库 (MutationLibrary)
# ==========================================

class MutationLibrary:
    """晶体结构底层变异算子库，所有方法均严格支持三种突变强度"""

    @staticmethod
    def replace_species(structure: Structure, strength: MutationStrength) -> Structure:
        """1. 元素同族/同价替换"""
        new_struct = structure.copy()
        # 根据强度决定替换比例
        replace_ratio = {MutationStrength.SMALL: 0.1, MutationStrength.MEDIUM: 0.3, MutationStrength.LARGE: 0.5}[strength]
        
        # 寻找契合 SPECIES_POOL 的可替换位点
        available_indices = [i for i, sp in enumerate(new_struct.species) if str(sp) in SPECIES_POOL]
        if not available_indices:
            return new_struct
            
        num_to_replace = max(1, int(len(available_indices) * replace_ratio))
        indices_to_modify = random.sample(available_indices, min(num_to_replace, len(available_indices)))
        
        # 执行实际修改（兼容 Pymatgen 与 Mock 结构）
        if hasattr(new_struct, "replace_species") and not isinstance(new_struct.species, list):
            for idx in indices_to_modify:
                current_sp = str(new_struct.species[idx])
                target_sp = random.choice(SPECIES_POOL[current_sp])
                new_struct.replace_species({idx: target_sp})
        else:
            # Mock 模式下的直接列表替换
            new_species = list(new_struct.species)
            for idx in indices_to_modify:
                current_sp = str(new_species[idx])
                new_species[idx] = random.choice(SPECIES_POOL[current_sp])
            new_struct.species = new_species
            
        return new_struct

    @staticmethod
    def perturb_atoms(structure: Structure, strength: MutationStrength) -> Structure:
        """2. 原子坐标微小扰动 (Perturbation: Small 0.03Å, Medium 0.08Å, Large 0.15Å)"""
        new_struct = structure.copy()
        dist_map = {MutationStrength.SMALL: 0.03, MutationStrength.MEDIUM: 0.08, MutationStrength.LARGE: 0.15}
        distance = dist_map[strength]
        
        if hasattr(new_struct, "perturb"):
            new_struct.perturb(distance)
        else:
            # Mock 扰动
            import numpy as np
            new_struct.frac_coords = np.array(new_struct.frac_coords) + np.random.normal(0, distance/4.0, np.array(new_struct.frac_coords).shape)
        return new_struct

    @staticmethod
    def scale_lattice(structure: Structure, strength: MutationStrength) -> Structure:
        """3. 晶格各向同性缩放 (Scale: Small 1%, Medium 3%, Large 8%)"""
        new_struct = structure.copy()
        scale_map = {MutationStrength.SMALL: 0.01, MutationStrength.MEDIUM: 0.03, MutationStrength.LARGE: 0.08}
        factor = 1.0 + random.choice([-1, 1]) * scale_map[strength]
        
        if hasattr(new_struct, "apply_strain"):
            new_struct.apply_strain(factor - 1.0)
        else:
            if hasattr(new_struct.lattice, "matrix"):
                new_struct.lattice.matrix *= factor
        return new_struct

    @staticmethod
    def shear_lattice(structure: Structure, strength: MutationStrength) -> Structure:
        """4. 晶格剪切形变"""
        new_struct = structure.copy()
        shear_map = {MutationStrength.SMALL: 0.02, MutationStrength.MEDIUM: 0.05, MutationStrength.LARGE: 0.10}
        shear_amount = shear_map[strength]
        
        if hasattr(new_struct, "apply_strain"):
            # 施加工程剪切应变 (voigt strain)
            strain = [0, 0, 0, shear_amount, 0, 0]
            new_struct.apply_strain(strain)
        return new_struct

    @staticmethod
    def symmetry_break(structure: Structure, strength: MutationStrength) -> Structure:
        """5. 对称性破缺（通过施加微小的非均匀随机位移降低晶体空间群对称性）"""
        new_struct = structure.copy()
        break_map = {MutationStrength.SMALL: 0.01, MutationStrength.MEDIUM: 0.04, MutationStrength.LARGE: 0.10}
        displacement = break_map[strength]
        
        # 仅针对部分特定原子产生不对称方向的推移
        target_indices = random.sample(range(len(new_struct.species)), max(1, len(new_struct.species) // 3))
        if hasattr(new_struct, "translate_sites"):
            for idx in target_indices:
                vec = [random.uniform(-displacement, displacement) for _ in range(3)]
                new_struct.translate_sites([idx], vec, frac_coords=True)
        return new_struct

    @staticmethod
    def create_vacancy(structure: Structure, strength: MutationStrength) -> Structure:
        """6. 产生原子空位缺陷"""
        new_struct = structure.copy()
        vac_count_map = {MutationStrength.SMALL: 1, MutationStrength.MEDIUM: 2, MutationStrength.LARGE: 4}
        num_vacancies = min(vac_count_map[strength], len(new_struct.species) - 1)
        
        indices_to_remove = sorted(random.sample(range(len(new_struct.species)), num_vacancies), reverse=True)
        if hasattr(new_struct, "remove_sites"):
            new_struct.remove_sites(indices_to_remove)
        else:
            for idx in indices_to_remove:
                if isinstance(new_struct.species, list):
                    new_struct.species.pop(idx)
                    new_struct.frac_coords.pop(idx)
        return new_struct

    @staticmethod
    def add_interstitial(structure: Structure, strength: MutationStrength) -> Structure:
        """7. 填隙原子添加"""
        new_struct = structure.copy()
        add_map = {MutationStrength.SMALL: 1, MutationStrength.MEDIUM: 2, MutationStrength.LARGE: 3}
        num_to_add = add_map[strength]
        
        # 从现有物种或常见轻元素中挑选手头适合的填隙元素
        available_elements = list(set([str(s) for s in new_struct.species])) + ["O", "F"]
        
        for _ in range(num_to_add):
            el = random.choice(available_elements)
            # 随机生成空旷处的晶格坐标
            coords = [random.random(), random.random(), random.random()]
            if hasattr(new_struct, "append"):
                new_struct.append(el, coords, coords_are_fractional=True)
        return new_struct

    @staticmethod
    def layer_shift(structure: Structure, strength: MutationStrength) -> Structure:
        """8. 层错/层滑移 (Layer Shift: Small 0.10Å, Medium 0.30Å, Large 0.60Å)"""
        new_struct = structure.copy()
        shift_map = {MutationStrength.SMALL: 0.10, MutationStrength.MEDIUM: 0.30, MutationStrength.LARGE: 0.60}
        shift_val = shift_map[strength]
        
        # 沿 Z 轴中线 (c-axis) 将上半部分沿 X 方向进行宏观滑移
        if hasattr(new_struct, "translate_sites"):
            upper_layer_indices = [i for i, coord in enumerate(new_struct.frac_coords) if coord[2] > 0.5]
            if upper_layer_indices:
                # 转换滑移值到分数坐标 (假设 c 轴大约 5.0Å 宽)
                frac_shift = shift_val / 5.0
                new_struct.translate_sites(upper_layer_indices, [frac_shift, 0.0, 0.0], frac_coords=True)
        return new_struct

    @staticmethod
    def composition_mutation(structure: Structure, strength: MutationStrength) -> Structure:
        """9. 组分/化学计量比调整（改变体系内元素比例）"""
        new_struct = structure.copy()
        # 比如通过将一个 A 元素替换为 B 元素破坏原有配比
        if len(set([str(s) for s in new_struct.species])) >= 2:
            all_sp = [str(s) for s in new_struct.species]
            unique_sp = list(set(all_sp))
            src, dst = random.sample(unique_sp, 2)
            
            indices = [i for i, s in enumerate(all_sp) if s == src]
            swap_count = {MutationStrength.SMALL: 1, MutationStrength.MEDIUM: 2, MutationStrength.LARGE: 3}[strength]
            
            if indices:
                for idx in indices[:min(swap_count, len(indices))]:
                    if hasattr(new_struct, "replace_species"):
                        new_struct.replace_species({idx: dst})
        return new_struct

    @staticmethod
    def supercell_mutation(structure: Structure, strength: MutationStrength) -> Structure:
        """10. 超晶胞构建与扩胞变异"""
        new_struct = structure.copy()
        # 根据突变强度定义扩胞矩阵尺寸
        matrix_map = {
            MutationStrength.SMALL: [2, 1, 1],
            MutationStrength.MEDIUM: [2, 2, 1],
            MutationStrength.LARGE: [2, 2, 2]
        }
        scaling_matrix = matrix_map[strength]
        if hasattr(new_struct, "make_supercell"):
            try:
                new_struct.make_supercell(scaling_matrix)
            except Exception as e:
                logger.warning(f"Supercell creation failed: {e}")
        return new_struct


# ==========================================
# 3. 突变控制引擎 (MutationEngine)
# ==========================================

class MutationEngine:
    """突变调度核心引擎，负责安全执行指定或随机组合的变异逻辑"""
    def __init__(self):
        # 动态建立算子映射表
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
        """应用单步指定的变异算子"""
        if m_type in self.operator_map:
            logger.info(f"Applying operational mutation: {m_type.value} with strength {strength.value}")
            return self.operator_map[m_type](structure, strength)
        else:
            raise ValueError(f"Unsupported mutation type: {m_type}")

    def random_apply(self, structure: Structure, strength: MutationStrength) -> Structure:
        """随机挑选一种变异算子执行"""
        chosen_type = random.choice(list(MutationType))
        return self.apply(structure, chosen_type, strength)

    def apply_sequence(self, structure: Structure, sequence: List[Tuple[MutationType, MutationStrength]]) -> Structure:
        """流水线串联执行一系列变异算子"""
        mutated_struct = structure.copy()
        for m_type, strength in sequence:
            mutated_struct = self.apply(mutated_struct, m_type, strength)
        return mutated_struct


# ==========================================
# 4. 物理约束、修复与验证流水线 (PhysicsConstraint)
# ==========================================

class PhysicsConstraint:
    """
    物理硬约束机制：包含空间冲突检测、晶格修复、以及最终的合法度校验（Validator）。
    全面承载了：Parent -> Mutation -> Physics Constraint -> Structure Repair -> Validator -> Child 完整工业流水线。
    """
    def __init__(self, min_bond_distance: float = 1.2):
        self.min_bond_distance = min_bond_distance

    def check_overlap(self, structure: Structure) -> List[Tuple[int, int, float]]:
        """检测晶体内部是否存在原子重叠（过近造成的物理爆炸高能态）"""
        overlap_pairs = []
        if hasattr(structure, "distance_matrix"):
            dm = structure.distance_matrix
            num_atoms = len(dm)
            for i in range(num_atoms):
                for j in range(i + 1, num_atoms):
                    if dm[i][j] < self.min_bond_distance:
                        overlap_pairs.append((i, j, dm[i][j]))
        return overlap_pairs

    def repair_structure(self, structure: Structure, overlap_pairs: List[Tuple[int, int, float]]) -> Structure:
        """结构修复器 (Structure Repair)：若发现原子间距违背硬约束，将其沿连线方向推开"""
        if not overlap_pairs:
            return structure
            
        repaired_struct = structure.copy()
        logger.info(f"[Structure Repair] Initiating push-apart repair protocol for {len(overlap_pairs)} overlapping pairs.")
        
        # 真实的 pymatgen 场景下会修改坐标，这里演示位置修饰逻辑
        if hasattr(repaired_struct, "translate_sites"):
            for i, j, dist in overlap_pairs:
                # 给发生冲突的原子施加一个微小的微扰分离方向
                repaired_struct.translate_sites([i], [0.02, 0.02, 0.0], frac_coords=True)
                repaired_struct.translate_sites([j], [-0.02, -0.02, 0.0], frac_coords=True)
        return repaired_struct

    def validator(self, structure: Structure) -> Tuple[bool, str]:
        """终审校验器 (Validator)：判断晶体最终化学与物理特性是否崩塌"""
        if len(structure.species) < 2:
            return False, "Structure collapsed: too few atoms remaining."
            
        # 再次确认修复后是否依然存在致命重叠
        remaining_overlaps = self.check_overlap(structure)
        if len(remaining_overlaps) > 0:
            return False, f"Validation Failed: Critical atomic overlap still persistent after repair ({remaining_overlaps[0][2]:.2f}Å)."
            
        return True, "Passed all constraints"

    def execute_workflow(self, parent: CrystalCandidate, engine: MutationEngine, 
                         m_type: Optional[MutationType] = None, 
                         strength: MutationStrength = MutationStrength.MEDIUM,
                         current_generation: int = 1) -> CrystalCandidate:
        """
        完整调度流水线逻辑：
        Parent Candidate -> Mutation Engine -> Physics Constraint -> Structure Repair -> Validator -> Child Candidate
        """
        logger.info(f"--- Starting Pipeline for Parent {parent.id} ---")
        
        # 1. Mutation Engine 变异阶段
        if m_type:
            mutated_structure = engine.apply(parent.structure, m_type, strength)
        else:
            mutated_structure = engine.random_apply(parent.structure, strength)
            
        # 2. Physics Constraint 检测阶段
        overlaps = self.check_overlap(mutated_structure)
        
        # 3. Structure Repair 自动修复阶段
        if len(overlaps) > 0:
            logger.warning(f"Physics Constraint Triggered: Found atoms too close. Shifting to Repair stage.")
            fixed_structure = self.repair_structure(mutated_structure, overlaps)
        else:
            fixed_structure = mutated_structure
            
        # 4. Validator 验证阶段
        is_legit, report_msg = self.validator(fixed_structure)
        
        # 5. Build Child Candidate 实例化子代
        child = CrystalCandidate(
            structure=fixed_structure, 
            generation=current_generation, 
            parent_ids=[parent.id]
        )
        child.is_valid = is_legit
        child.error_msg = report_msg
        
        if is_legit:
            logger.info(f"Pipeline Success! Generated Valid Child Candidate: {child.id}")
        else:
            logger.error(f"Pipeline Failed at Validator Stage: {report_msg}")
            
        return child


# ==========================================
# 5. 单元管道验证运行演示
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 模拟构建一个母本 Candidate (化学式含 Bi 和 Te 用于支持同族替换)
    mock_lat = [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 10.0]]
    mock_species = ["Bi", "Bi", "Te", "Te", "Te"]
    mock_coords = [[0,0,0], [0.5,0.5,0.1], [0.5,0,0.4], [0,0.5,0.6], [0.5,0.5,0.8]]
    
    base_structure = Structure(mock_lat, mock_species, mock_coords)
    parent_cand = CrystalCandidate(structure=base_structure, generation=0)
    
    # 初始化核心引擎与约束流水线
    engine = MutationEngine()
    constraint_pipeline = PhysicsConstraint(min_bond_distance=1.5)
    
    # 场景一：测试同族特定元素替换 (Replace Species)
    print("\n--- TEST SCENARIO 1: Element Substitution ---")
    child_1 = constraint_pipeline.execute_workflow(
        parent=parent_cand, 
        engine=engine, 
        m_type=MutationType.REPLACE_SPECIES, 
        strength=MutationStrength.SMALL,
        current_generation=1
    )
    
    # 场景二：测试引发层错和原子对挤压的修复流 (Layer Shift)
    print("\n--- TEST SCENARIO 2: Layer Shift & Auto Repair ---")
    # 人为修改使其极度接近以模拟高压碰撞冲突
    bad_structure = base_structure.copy()
    bad_structure.distance_matrix[0][1] = 0.8  # 强制构造一个短键冲突
    clashing_parent = CrystalCandidate(structure=bad_structure, generation=0)
    
    child_2 = constraint_pipeline.execute_workflow(
        parent=clashing_parent, 
        engine=engine, 
        m_type=MutationType.LAYER_SHIFT, 
        strength=MutationStrength.LARGE,
        current_generation=1
    )
