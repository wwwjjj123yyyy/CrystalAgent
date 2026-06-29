import os
import random
import json
import numpy as np
from typing import List, Dict, Any, Tuple

# 模拟或引入材料科学核心库 (如 pymatgen)
# 在实际环境中，请确保安装了 pymatgen: pip install pymatgen
try:
    from pymatgen.core import Structure, Lattice
except ImportError:
    # 如果没有安装，这里提供一个极简的 Mock 类保证代码可运行
    class Lattice:
        @classmethod
        def cubic(cls, a): return {"a": a, "type": "cubic"}
    class Structure:
        def __init__(self, lattice, species, coords):
            self.lattice = lattice
            self.species = species
            self.coords = coords
        def formula(self): return "".join([f"{k}{v}" for k,v in self.composition.items()])
        @property
        def composition(self): return {s: self.species.count(s) for s in set(self.species)}


# ==========================================
# 0. 数据结构定义 (Core Data Structures)
# ==========================================

class CrystalCandidate:
    """晶体结构候选者类，记录其基因、物理结构及评估指标"""
    def __init__(self, structure: Structure, generation: int, parent_ids: List[str] = None):
        self.id = f"cand_{generation}_{random.randint(1000, 9999)}"
        self.structure = structure
        self.generation = generation
        self.parent_ids = parent_ids or []
        
        # 核心优化目标
        self.stability_score: float = 0.0      # 稳定性 (如: -Energy above hull)
        self.comp_novelty_score: float = 0.0   # 成分新颖性
        self.struct_novelty_score: float = 0.0 # 结构新颖性
        self.fitness: float = 0.0              # 综合适应度
        
        # 物理与验证状态
        self.is_valid: bool = False
        self.error_msg: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "formula": self.structure.formula if hasattr(self.structure, 'formula') else "MockFormula",
            "fitness": self.fitness,
            "metrics": {
                "stability": self.stability_score,
                "composition_novelty": self.comp_novelty_score,
                "structure_novelty": self.struct_novelty_score
            },
            "generation": self.generation
        }


# ==========================================
# 1. 初始化阶段 (Initialization Phase)
# ==========================================

class SeedInitializer:
    """种子初始化器：从 Materials Project 等数据库加载已知稳定结构作为演化起点"""
    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def fetch_mp_seeds(self, criteria: Dict[str, Any], limit: int = 10) -> List[Structure]:
        """模拟从 Materials Project 抓取优质种子"""
        print(f"[SeedInitializer] Fetching {limit} seeds from Materials Project database...")
        # 实际开发中这里会使用 MPRester(self.api_key) 搜索符合条件的稳定结构
        # 这里使用 Mock 数据作为示例
        mock_seeds = []
        for i in range(limit):
            lat = Lattice.cubic(4.0 + random.random())
            # 随机生成一些基础氧化物/三元化合物作为种子
            species = ["Sr", "Ti", "O", "O", "O"] if i % 2 == 0 else ["Ba", "Fe", "O", "O", "O"]
            coords = [[0,0,0], [0.5,0.5,0.5], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]
            mock_seeds.append(Structure(lat, species, coords))
        return mock_seeds

    def initialize_population(self, size: int) -> List[CrystalCandidate]:
        """构建初始种群 (Generation 0)"""
        seeds = self.fetch_mp_seeds(criteria={"energy_above_hull": 0.0}, limit=size)
        population = [CrystalCandidate(struct, generation=0) for struct in seeds]
        return population


# ==========================================
# 2. 演化循环阶段 (Evolution Loop Components)
# ==========================================

class LLMPlanner:
    """Agent 决策核心：Planner (LLM) 负责生成全局演化策略与操作指令"""
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name

    def plan_next_generation(self, generation: int, memory: Dict[str, Any], elites: List[CrystalCandidate]) -> List[Dict[str, Any]]:
        """
        基于当前历史记忆 (Memory) 和 精英个体 (Elites)，让 LLM 决定本轮演化操作。
        返回一组演化操作指令 (Mutation/Crossover 计划)。
        """
        print(f"\n[Planner (LLM)] Analyzing Generation {generation} performance...")
        
        # 构建 Prompt 上下文
        elite_summaries = [e.to_dict() for e in elites[:3]]
        prompt = f"""
        Current Generation: {generation}
        Past Successes/Failures: {json.dumps(memory.get('reflection_history', []))}
        Current Top Elites: {json.dumps(elite_summaries)}
        
        Task: Propose an evolutionary strategy to balance Stability, Composition Novelty, and Structure Novelty.
        Output a list of operation JSON instructions for the Evolution Library.
        Available operations: 'element_substitution', 'lattice_distortion', 'stoichiometry_shift'.
        """
        
        # 模拟 LLM API 呼叫与解析过程
        # response = openai.ChatCompletion.create(model=self.model_name, messages=[...])
        print(f"[Planner (LLM)] Thinking... Formulating mutation strategies.")
        
        # Mock LLM 返回的结构化指令
        mock_instructions = [
            {"operation": "element_substitution", "target_idx": 0, "params": {"replace_dict": {"Sr": "Ca", "Ba": "Mg"}}},
            {"operation": "lattice_distortion", "target_idx": 1, "params": {"strain_tensor": [0.05, 0.0, 0.0]}},
            {"operation": "stoichiometry_shift", "target_idx": 0, "params": {"add_element": "F"}}
        ]
        return mock_instructions


class EvolutionLibrary:
    """演化算子库：执行具体的结构层面的变异 (Mutation) 和交叉 (Crossover)"""
    @staticmethod
    def mutate_substitution(structure: Structure, replace_dict: Dict[str, str]) -> Structure:
        """元素替换"""
        # 实际代码中会调用 structure.replace_species(replace_dict)
        new_species = [replace_dict.get(sp, sp) for sp in structure.species]
        return Structure(structure.lattice, new_species, structure.coords)

    @staticmethod
    def mutate_distortion(structure: Structure, strain_tensor: List[float]) -> Structure:
        """晶格畸变操作"""
        # 实际会应用应变到张量上
        return structure 

    @staticmethod
    def crossover(parent1: Structure, parent2: Structure) -> Structure:
        """切割拼接式交叉 (Cut-and-Splice Crossover)"""
        # 结合双亲的晶胞与坐标
        return parent1


class StructureBuilder:
    """结构构建器：将 Planner 的宏观意图与 EvolutionLibrary 的算子结合，实例化新结构"""
    def __init__(self, library: EvolutionLibrary):
        self.lib = library

    def build_candidates(self, instructions: List[Dict[str, Any]], pool: List[CrystalCandidate], gen: int) -> List[CrystalCandidate]:
        print("[Structure Builder] Translating LLM instructions into 3D crystal structures...")
        new_candidates = []
        
        for inst in instructions:
            if not pool: break
            # 随机选择或根据 target_idx 选择父代
            parent = pool[inst["target_idx"] % len(pool)]
            op = inst["operation"]
            params = inst["params"]
            
            try:
                if op == "element_substitution":
                    new_struct = self.lib.mutate_substitution(parent.structure, params["replace_dict"])
                elif op == "lattice_distortion":
                    new_struct = self.lib.mutate_distortion(parent.structure, params["strain_tensor"])
                else:
                    new_struct = parent.structure # Fallback
                
                child = CrystalCandidate(new_struct, generation=gen, parent_ids=[parent.id])
                new_candidates.append(child)
            except Exception as e:
                print(f"Failed to build structure via {op}: {e}")
                
        return new_candidates


class PhysicsValidator:
    """物理约束校验器：对生成的结构进行硬性物理过滤，剔除不合理的毒结构"""
    def __init__(self):
        pass

    def validate(self, candidate: CrystalCandidate) -> bool:
        # 1. 检查原子间距是否过近 (原子重叠)
        # distance_matrix = candidate.structure.distance_matrix
        
        # 2. 检查基本的电荷平衡 (Electronegativity / Charge Balance)
        
        # 3. 晶体空间群及对称性检查
        
        # Mock 校验逻辑：大语言模型生成的策略有 90% 概率通过基础物理约束
        is_valid = random.random() > 0.1
        candidate.is_valid = is_valid
        if not is_valid:
            candidate.error_msg = "Atomic distance too close (overlap detected)."
        return is_valid


class IndependentEvaluator:
    """独立评估器：并行计算稳定性、成分新颖性、结构新颖性三大底层指标"""
    def __init__(self, reference_db: List[Structure] = None):
        self.reference_db = reference_db or []

    def evaluate_stability(self, candidate: CrystalCandidate) -> float:
        """评估稳定性：通过预训练机器学习势能模型 (如 M3GNet, CHGNet) 预测 Energy Above Hull"""
        # 越接近或低于 0 越稳定，我们返回一个越高越好的得分
        predicted_energy = -1.0 * random.uniform(0.0, 0.5) 
        return float(np.exp(predicted_energy)) # 映射到 (0, 1] 

    def evaluate_composition_novelty(self, candidate: CrystalCandidate) -> float:
        """评估成分新颖性：检查该化学体系是否在已知数据库(ICSD/MP)中出现过"""
        # 计算该成分在参考库里的频率
        return random.uniform(0.4, 0.9)

    def evaluate_structure_novelty(self, candidate: CrystalCandidate) -> float:
        """评估结构新颖性：基于晶体指纹 (Crystal Fingerprint) 或 XRD 相似度进行对比"""
        # 相似度越低，新颖性越高
        return random.uniform(0.5, 0.95)

    def process(self, candidate: CrystalCandidate):
        """全面测定指标"""
        candidate.stability_score = self.evaluate_stability(candidate)
        candidate.comp_novelty_score = self.evaluate_composition_novelty(candidate)
        candidate.struct_novelty_score = self.evaluate_structure_novelty(candidate)


class FitnessCalculator:
    """适应度计算器：将多目标优化转换为标量 Fitness，或计算帕累托前沿 (Pareto Frontier)"""
    def __init__(self, weights: Dict[str, float] = None):
        # 默认三者权重平衡
        self.weights = weights or {"stability": 0.4, "comp_novelty": 0.3, "struct_novelty": 0.3}

    def calculate(self, candidate: CrystalCandidate) -> float:
        if not candidate.is_valid:
            candidate.fitness = 0.0
            return 0.0
        
        # 线性加权组合
        f = (self.weights["stability"] * candidate.stability_score +
             self.weights["comp_novelty"] * candidate.comp_novelty_score +
             self.weights["struct_novelty"] * candidate.struct_novelty_score)
        candidate.fitness = float(f)
        return candidate.fitness


class EliteSelection:
    """精英选择器：保留最优秀的个体，淘汰低适应度结构，并作为下一代繁衍的池子"""
    def __init__(self, elite_ratio: float = 0.2):
        self.elite_ratio = elite_ratio

    def select(self, population: List[CrystalCandidate], keep_size: int) -> Tuple[List[CrystalCandidate], List[CrystalCandidate]]:
        # 按 Fitness 降序排序
        sorted_pop = sorted([c for c in population if c.is_valid], key=lambda x: x.fitness, reverse=True)
        
        elite_count = max(1, int(keep_size * self.elite_ratio))
        elites = sorted_pop[:elite_count]
        survivors = sorted_pop[:keep_size] # 允许进入下一代交叉变异池的结构
        
        return elites, survivors


# ==========================================
# 3. 智能体闭环与记忆 (Reflection & Memory)
# ==========================================

class AgentMemorySystem:
    """Agent 记忆与反思系统：存储历史运行数据，通过 LLM 反思优化后续搜索效率"""
    def __init__(self):
        self.memory_space = {
            "all_evaluated_count": 0,
            "failed_constraints_count": 0,
            "reflection_history": [],
            "best_fitness_progress": []
        }

    def update(self, generation: int, current_pop: List[CrystalCandidate], invalid_count: int):
        self.memory_space["all_evaluated_count"] += len(current_pop)
        self.memory_space["failed_constraints_count"] += invalid_count
        
        valid_fitnesses = [c.fitness for c in current_pop if c.is_valid]
        max_fit = max(valid_fitnesses) if valid_fitnesses else 0.0
        self.memory_space["best_fitness_progress"].append({"gen": generation, "max_fitness": max_fit})

    def reflect(self, generation: int, elites: List[CrystalCandidate]):
        """Agent 反思机制：分析本代精英，提炼出哪些元素组合或结构特征带来了‘高新颖性’与‘高稳定性’"""
        print(f"[Reflection] Agent reflecting on Generation {generation} achievements...")
        
        # 在真实场景下，将数据打包喂给 LLM 
        # "Why did these structures survive? Because substituting Sr with Ca retained stability while boosting novelty."
        mock_reflection = f"Gen {generation}: Element substitution with Ca showed high stability. Keep exploring alkaline earth metals."
        self.memory_space["reflection_history"].append(mock_reflection)


# ==========================================
# 4. 主流程编排引擎 (Main Execution Engine)
# ==========================================

class CrystalEvolutionEngine:
    def __init__(self, pop_size: int = 10, max_generations: int = 3):
        self.pop_size = pop_size
        self.max_generations = max_generations
        
        # 组件实例化
        self.initializer = SeedInitializer()
        self.planner = LLMPlanner()
        self.builder = StructureBuilder(library=EvolutionLibrary())
        self.validator = PhysicsValidator()
        self.evaluator = IndependentEvaluator()
        self.fitness_calc = FitnessCalculator()
        self.selector = EliteSelection()
        self.memory_sys = AgentMemorySystem()

    def run_discovery_pipeline(self):
        print("="*60)
        print("  STARTING LLM + EVOLUTIONARY CRYSTAL SEARCH PIPELINE  ")
        print("="*60)

        # Step 1: Seed Initializer & Initial Population
        population = self.initializer.initialize_population(size=self.pop_size)
        
        # 初始评估
        for cand in population:
            cand.is_valid = True # 种子默认合法
            self.evaluator.process(cand)
            self.fitness_calc.calculate(cand)
            
        elites = population[:2] # 初始假定前两个为精英

        # Step 2: Evolution Loop
        for gen in range(1, self.max_generations + 1):
            print(f"\n--- [GENERATION {gen} / {self.max_generations}] ---")
            
            # 2.1 Planner (LLM) 生成本代动作策略
            instructions = self.planner.plan_next_generation(gen, self.memory_sys.memory_space, elites)
            
            # 2.2 Structure Builder
            new_candidates = self.builder.build_candidates(instructions, population, gen)
            
            # 2.3 & 2.4 Physics Validator & Independent Evaluator & Fitness Calculator
            invalid_this_gen = 0
            for cand in new_candidates:
                # 物理过滤
                if not self.validator.validate(cand):
                    invalid_this_gen += 1
                    continue
                
                # 性能多指标并行度量
                self.evaluator.process(cand)
                # 计算总评分
                self.fitness_calc.calculate(cand)
            
            # 合并父代和新子代进行合并筛选 (Lambda + Mu 策略)
            combined_population = population + new_candidates
            
            # 2.5 Elite Selection
            elites, population = self.selector.select(combined_population, keep_size=self.pop_size)
            
            print(f"[Evolution Loop] Filtered: {invalid_this_gen} failed constraints. Top Fitness: {elites[0].fitness:.4f}")
            
            # 2.6 Reflection
            self.memory_sys.reflect(gen, elites)
            
            # 2.7 Memory Update
            self.memory_sys.update(gen, population, invalid_this_gen)
            
        print("\n" + "="*40)
        print("  DISCOVERY PIPELINE COMPLETED  ")
        print("="*40)
        print(f"Top discovered structural candidates written to logs. Total evaluated: {self.memory_sys.memory_space['all_evaluated_count']}\n")
        for idx, elite in enumerate(elites[:3]):
            print(f"Elite {idx+1}: ID={elite.id} | Fitness={elite.fitness:.4f} | Stability={elite.stability_score:.3f} | CompNovelty={elite.comp_novelty_score:.3f} | StructNovelty={elite.struct_novelty_score:.3f}")


if __name__ == "__main__":
    # 执行搜索闭环
    engine = CrystalEvolutionEngine(pop_size=5, max_generations=3)
    engine.run_discovery_pipeline()
