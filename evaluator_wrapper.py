import logging
import random
import math
from typing import Dict, Any

# 引入或兼容 Pymatgen 核心库
try:
    from pymatgen.core import Structure
except ImportError:
    class Structure:
        def __init__(self, lattice, species, coords):
            self.lattice = lattice
            self.species = species
            self.frac_coords = coords
        @property
        def formula(self) -> str:
            return "Bi2Te3"

# 引入核心 Candidate 数据结构
try:
    from core.candidate import CrystalCandidate
except ImportError:
    from typing import TypeVar
    CrystalCandidate = TypeVar('CrystalCandidate')

# 配置日志
logger = logging.getLogger("EvaluatorWrapper")
logger.setLevel(logging.INFO)


class EvaluatorWrapper:
    """
    评估器包装器 (Evaluator Wrapper)
    负责多维度评估晶体结构，并计算出最终用于指导遗传选择的综合适应度（Fitness）。
    """
    def __init__(
        self, 
        w_energy: float = 0.6, 
        w_comp_novelty: float = 0.2, 
        w_struct_novelty: float = 0.2
    ):
        """
        Args:
            w_energy: 能量稳定性的权重（通常占比最高，保证物理可合成性）
            w_comp_novelty: 化学成分新颖性的权重（鼓励 Agent 探索未知领域）
            w_struct_novelty: 空间拓扑结构新颖性的权重（鼓励打破常规晶格）
        """
        self.w_energy = w_energy
        self.w_comp_novelty = w_comp_novelty
        self.w_struct_novelty = w_struct_novelty
        
        # 归一化权重确认
        total_w = w_energy + w_comp_novelty + w_struct_novelty
        if not math.isclose(total_w, 1.0):
            logger.warning(f"Weights do not sum up to 1.0 (Sum: {total_w}). Automatically normalizing weights.")
            self.w_energy /= total_w
            self.w_comp_novelty /= total_w
            self.w_struct_novelty /= total_w

    def _evaluate_energy_stability(self, structure: Structure) -> float:
        """
        核心 1：评估能量稳定性分数。
        真实生产中，在此处导入机器学习势能模型（如 CHGNet, M3GNet）或调用 VASP。
        返回一个归一化的分数 [0, 1]，分数越高代表越趋向于热力学基态（越稳定）。
        """
        try:
            # -----------------------------------------------------------------
            # [生产集成示例插槽]
            # from chgnet.model.model import CHGNet
            # chgnet = CHGNet.load()
            # prediction = chgnet.predict_structure(structure)
            # energy_per_atom = float(prediction["energy_per_atom"])
            # -----------------------------------------------------------------
            
            # 此处使用模拟数据：通常晶体每个原子的能量在 -8 eV 到 -2 eV 之间
            # 我们将其映射为 0~1 的稳定度得分
            simulated_energy = random.uniform(-7.5, -3.0)
            
            # 映射公式：能量越负，得分越接近 1
            stability_score = 1.0 - (simulated_energy - (-8.0)) / (-2.0 - (-8.0))
            return max(0.0, min(1.0, stability_score))
            
        except Exception as e:
            logger.error(f"Error evaluating ML energy: {e}")
            return 0.0

    def _evaluate_composition_novelty(self, structure: Structure) -> float:
        """
        核心 2：评估化学成分新颖性分数 [0, 1]。
        可以与外部的已合成晶体知识库（如 Materials Project 已有条目）对比。
        如果当前化学式在已知库里从未出现过，打高分。
        """
        formula = getattr(structure, "formula", "Unknown")
        
        # 模拟外部数据库查询过滤
        known_formulas = ["Bi2Te3", "Sb2Te3", "Bi2Se3", "TiO2", "SrTiO3"]
        
        if formula in known_formulas:
            # 已知结构，新颖性得分较低
            return 0.2
        else:
            # 未知新型化学配比，给予高分鼓励
            return 0.95

    def _evaluate_structure_novelty(self, structure: Structure) -> float:
        """
        核心 3：评估空间结构/几何新颖性分数 [0, 1]。
        可以通过比较晶胞的 RDF（径向分布函数）或空间群（Space Group）来计算指纹相似度。
        """
        # 此处模拟返回结构畸变与非对称带来的新颖性得分
        return round(random.uniform(0.4, 0.85), 4)

    def process(self, candidate: CrystalCandidate) -> float:
        """
        流水线主入口：执行多指标评测，融合并更新 Candidate 的 fitness。
        
        Args:
            candidate: 待评估的晶体候选者（必须已通过 Validator 验证）
            
        Returns:
            float: 最终计算出的综合 Fitness
        """
        if not candidate.is_valid:
            logger.warning(f"Candidate {candidate.id} is flagged as INVALID. Skipping evaluation.")
            candidate.fitness = 0.0
            return 0.0

        logger.info(f"Evaluating candidate {candidate.id} (Formula: {getattr(candidate.structure, 'formula', 'Unknown')})...")

        # 1. 分别计算三大指标
        s_energy = self._evaluate_energy_stability(candidate.structure)
        s_comp = self._evaluate_composition_novelty(candidate.structure)
        s_struct = self._evaluate_structure_novelty(candidate.structure)

        # 2. 将细分指标挂载到 candidate 身上，方便 population 和 planner 读取反思
        if not hasattr(candidate, "metrics"):
            candidate.metrics = {}
        
        candidate.metrics["stability"] = s_energy
        candidate.metrics["composition_novelty"] = s_comp
        candidate.metrics["structure_novelty"] = s_struct

        # 3. 线性加权融合计算最终的 Fitness
        final_fitness = (
            self.w_energy * s_energy +
            self.w_comp_novelty * s_comp +
            self.w_struct_novelty * s_struct
        )
        
        # 4. 回填赋值
        candidate.fitness = round(final_fitness, 4)
        
        logger.info(f"Evaluation complete for {candidate.id}: Fitness = {candidate.fitness} (Energy: {s_energy:.2f}, CompNovelty: {s_comp:.2f}, StructNovelty: {s_struct:.2f})")
        
        return candidate.fitness


# ==========================================
# 单元测试与演示
# ==========================================
if __name__ == "__main__":
    print("--- Testing EvaluatorWrapper Component ---")
    
    # 建立模拟合法的 Candidate
    good_struct = Structure(None, ["Bi", "Te"], [[0,0,0], [0.5,0.5,0.5]])
    
    class MockCandidate:
        def __init__(self, structure):
            self.id = "cand_8888"
            self.structure = structure
            self.is_valid = True
            self.fitness = 0.0
            self.metrics = {}

    candidate = MockCandidate(good_struct)

    # 初始化评估器，分配权重为 能量(60%)、成分新颖度(20%)、结构新颖度(20%)
    evaluator = EvaluatorWrapper(w_energy=0.6, w_comp_novelty=0.2, w_struct_novelty=0.2)
    
    # 运行评估
    fitness_score = evaluator.process(candidate)
    print(f"\nResulting Metadata in Candidate:")
    print(f"Overall Fitness: {candidate.fitness}")
    print(f"Detailed Metrics: {candidate.metrics}")
