import json
import logging
from collections import defaultdict
import random
from typing import List, Dict, Any, Set, Optional

# 假定 candidate.py 在同一目录下
# 为了保证脚本可独立运行，如果导入失败则使用基础提示
try:
    from candidate import CrystalCandidate, Structure, Lattice
except ImportError:
    # 兼容未拆分文件时的 Mock 类
    from typing import TypeVar
    CrystalCandidate = TypeVar('CrystalCandidate')
    Structure = TypeVar('Structure')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PopulationManager")


class PopulationManager:
    """
    晶体种群管理器 (Population Manager)
    负责统一管理演化过程中的所有 CrystalCandidate，维护四大核心池，
    提供高效的公式/世代索引、多样性过滤采样以及数据持久化接口。
    """
    def __init__(self):
        # 1. 四大核心候选池
        self.active_pool: List[CrystalCandidate] = []   # 当前代处于激活/繁衍状态的种群
        self.elite_pool: List[CrystalCandidate] = []    # 历代最顶尖的精英个体
        self.archive_pool: List[CrystalCandidate] = []  # 历史产生的所有合法结构大本营 (知识库)
        self.failed_pool: List[CrystalCandidate] = []   # 未通过物理约束或严重不稳定的失效结构

        # 2. 高效索引表 (加速 Planner 和 Evaluator 查询)
        self.formula_index: Dict[str, List[CrystalCandidate]] = defaultdict(list)
        self.generation_index: Dict[int, List[CrystalCandidate]] = defaultdict(list)

    # ==========================================
    # 池管理与索引核心操作
    # ==========================================

    def add_candidate(self, candidate: CrystalCandidate, pool_name: str = "active") -> None:
        """
        向指定的池中添加新的晶体候选者，并自动更新全局索引
        """
        pool_name = pool_name.lower()
        if pool_name == "active":
            self.active_pool.append(candidate)
        elif pool_name == "elite":
            self.elite_pool.append(candidate)
        elif pool_name == "archive":
            self.archive_pool.append(candidate)
        elif pool_name == "failed":
            self.failed_pool.append(candidate)
        else:
            logger.error(f"Unknown pool name: {pool_name}. Falling back to active_pool.")
            self.active_pool.append(candidate)

        # 维护索引
        self._update_indices(candidate)
        logger.debug(f"Candidate {candidate.id} added to {pool_name} pool.")

    def _update_indices(self, candidate: CrystalCandidate) -> None:
        """更新化学式和世代索引"""
        # 获取化学式
        formula = "Unknown"
        if hasattr(candidate.structure, "formula"):
            formula = candidate.structure.formula
        elif hasattr(candidate, "formula"):
            formula = candidate.formula
        elif isinstance(candidate.structure, dict):
            formula = candidate.structure.get("formula", "Unknown")

        # 建立双向映射
        if candidate not in self.formula_index[formula]:
            self.formula_index[formula].append(candidate)
        if candidate not in self.generation_index[candidate.generation]:
            self.generation_index[candidate.generation].append(candidate)

    def get_by_formula(self, formula: str) -> List[CrystalCandidate]:
        """通过化学成分公式精确查找候选结构 (如 'SrTiO3')"""
        return self.formula_index.get(formula, [])

    def get_by_generation(self, generation: int) -> List[CrystalCandidate]:
        """获取某一特定世代产生的全部候选结构"""
        return self.generation_index.get(generation, [])

    # ==========================================
    # 排序与精英更新机制
    # ==========================================

    def sort_pool_by_fitness(self, pool_name: str = "active") -> List[CrystalCandidate]:
        """按适应度（Fitness）从高到低对指定池进行排序"""
        pool_map = {
            "active": self.active_pool,
            "elite": self.elite_pool,
            "archive": self.archive_pool,
            "failed": self.failed_pool
        }
        target_pool = pool_map.get(pool_name.lower(), self.active_pool)
        # 降序排序
        target_pool.sort(key=lambda x: x.fitness, reverse=True)
        return target_pool

    def refresh_elite_pool(self, max_elite_size: int = 10) -> None:
        """
        根据全局最优规则，从 active 和 archive 池中筛选真正的顶尖个体更新精英池
        """
        # 合并所有合法的历史与当前结构
        all_valid_candidates = self.active_pool + self.archive_pool
        # 去重（按 ID）
        unique_candidates = {c.id: c for c in all_valid_candidates if c.is_valid}.values()
        
        # 排序并取前 N 个
        sorted_candidates = sorted(unique_candidates, key=lambda x: x.fitness, reverse=True)
        self.elite_pool = sorted_candidates[:max_elite_size]
        logger.info(f"Elite pool refreshed. Current top fitness: {self.elite_pool[0].fitness if self.elite_pool else 0.0:.4f}")

    # ==========================================
    # 多样性采样算法 (Diversity Sampling)
    # ==========================================

    def sample_with_diversity(self, pool_name: str = "active", sample_size: int = 5) -> List[CrystalCandidate]:
        """
        贪婪多样性采样算法：
        为了避免 LLM 或演化算法陷入单一化学体系（局部最优），
        该采样策略优先选择『适应度高』且『化学成分不重复』的晶体结构。
        """
        pool_map = {"active": self.active_pool, "archive": self.archive_pool, "elite": self.elite_pool}
        source_pool = pool_map.get(pool_name.lower(), self.active_pool)
        
        if len(source_pool) <= sample_size:
            return source_pool.copy()

        # 先按适应度降序排序
        sorted_pool = sorted(source_pool, key=lambda x: x.fitness, reverse=True)
        
        sampled_candidates: List[CrystalCandidate] = []
        selected_formulas: Set[str] = set()

        # 第一阶段：尝试挑选化学式完全不重复的优秀结构
        for cand in sorted_pool:
            formula = getattr(cand.structure, "formula", "Unknown")
            if formula not in selected_formulas:
                sampled_candidates.append(cand)
                selected_formulas.add(formula)
            if len(sampled_candidates) == sample_size:
                break

        # 第二阶段：如果独特化学式数量不够，用剩余的高适应度结构填满
        if len(sampled_candidates) < sample_size:
            for cand in sorted_pool:
                if cand not in sampled_candidates:
                    sampled_candidates.append(cand)
                if len(sampled_candidates) == sample_size:
                    break

        logger.info(f"Sampled {len(sampled_candidates)} candidates from {pool_name} with diversity constraints.")
        return sampled_candidates

    # ==========================================
    # 统计接口 (Statistical Interface)
    # ==========================================

    def get_statistics(self) -> Dict[str, Any]:
        """
        分析当前整个系统的运行状态，输出多维度的统计快照。
        Planner (LLM) 可以直接读取该统计字典作为反思（Reflection）的输入。
        """
        def safe_max_avg(pool: List[CrystalCandidate]):
            if not pool: return 0.0, 0.0
            fits = [c.fitness for c in pool]
            return max(fits), sum(fits) / len(fits)

        active_max, active_avg = safe_max_avg(self.active_pool)
        archive_max, archive_avg = safe_max_avg(self.archive_pool)

        stats = {
            "pools_size": {
                "active": len(self.active_pool),
                "elite": len(self.elite_pool),
                "archive": len(self.archive_pool),
                "failed": len(self.failed_pool),
                "total_discovered": len(self.active_pool) + len(self.archive_pool) + len(self.failed_pool)
            },
            "fitness_metrics": {
                "active_max_fitness": active_max,
                "active_avg_fitness": active_avg,
                "global_best_fitness": self.elite_pool[0].fitness if self.elite_pool else 0.0,
            },
            "diversity_metrics": {
                "unique_formulas_count": len(self.formula_index),
                "tracked_generations": list(self.generation_index.keys())
            }
        }
        return stats

    # ==========================================
    # JSON 持久化 (Serialization & Deserialization)
    # ==========================================

    def save_to_json(self, filepath: str) -> None:
        """
        将四大池的完整状态与结构序列化保存为 JSON 文件，供随时中断与复盘。
        """
        data_to_save = {
            "active_pool": [c.to_dict() for c in self.active_pool],
            "elite_pool": [c.to_dict() for c in self.elite_pool],
            "archive_pool": [c.to_dict() for c in self.archive_pool],
            "failed_pool": [
                {**c.to_dict(), "error_msg": getattr(c, "error_msg", "Unknown error")} 
                for c in self.failed_pool
            ]
        }
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=4, ensure_ascii=False)
            logger.info(f"Successfully serialized and saved population state to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save population to JSON: {e}")

    def load_from_json(self, filepath: str) -> None:
        """
        从 JSON 备份文件中恢复持久化的种群数据，并重新建立内存索引。
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 清空当前内存状态
            self.__init__()

            def reconstruct_pool(json_list: List[Dict[str, Any]], pool_name: str):
                for item in json_list:
                    # 1. 恢复基础的 3D 晶体 mock 结构 (实际中通过还原 Lattice/Species 实例化)
                    # 这里为了兼容性，用一个带有 formula 的通用对象
                    class MockStructure:
                        def __init__(self, formula): self.formula = formula
                    
                    struct_mock = MockStructure(item.get("formula", "Unknown"))
                    
                    # 2. 实例化 Candidate
                    cand = CrystalCandidate(
                        structure=struct_mock, 
                        generation=item.get("generation", 0)
                    )
                    cand.id = item.get("id", cand.id)
                    cand.fitness = item.get("fitness", 0.0)
                    
                    metrics = item.get("metrics", {})
                    cand.stability_score = metrics.get("stability", 0.0)
                    cand.comp_novelty_score = metrics.get("composition_novelty", 0.0)
                    cand.struct_novelty_score = metrics.get("structure_novelty", 0.0)
                    cand.is_valid = True if pool_name != "failed" else False
                    if pool_name == "failed":
                        cand.error_msg = item.get("error_msg", "")

                    # 3. 归入对应的池
                    self.add_candidate(cand, pool_name=pool_name)

            reconstruct_pool(data.get("active_pool", []), "active")
            reconstruct_pool(data.get("elite_pool", []), "elite")
            reconstruct_pool(data.get("archive_pool", []), "archive")
            reconstruct_pool(data.get("failed_pool", []), "failed")
            
            logger.info(f"Successfully loaded population state from {filepath}. Total loaded formulas: {len(self.formula_index)}")
        except Exception as e:
            logger.error(f"Failed to load population from JSON: {e}")


# ==========================================
# 5. 单元测试与用例演示
# ==========================================
if __name__ == "__main__":
    from candidate import CrystalCandidate
    
    print("--- Testing PopulationManager Implementation ---")
    manager = PopulationManager()
    
    # 模拟构建几个来自不同世代、不同成分的晶体
    class DummyStructure:
        def __init__(self, formula): self.formula = formula

    # 制造候选结构
    c1 = CrystalCandidate(DummyStructure("SrTiO3"), generation=1)
    c1.fitness, c1.is_valid = 0.85, True
    
    c2 = CrystalCandidate(DummyStructure("BaTiO3"), generation=1)
    c2.fitness, c2.is_valid = 0.92, True
    
    c3 = CrystalCandidate(DummyStructure("SrTiO3"), generation=2) # 同化学式，不同代
    c3.fitness, c3.is_valid = 0.74, True
    
    c4 = CrystalCandidate(DummyStructure("Fe2O3"), generation=2)
    c4.fitness, c4.is_valid = 0.95, True
    
    c5 = CrystalCandidate(DummyStructure("UUU_Broken"), generation=2) # 违背物理约束的结构
    c5.fitness, c5.is_valid, c5.error_msg = 0.0, False, "Exploded cell"

    # 将它们推进对应的池子
    manager.add_candidate(c1, "active")
    manager.add_candidate(c2, "active")
    manager.add_candidate(c3, "archive")
    manager.add_candidate(c4, "active")
    manager.add_candidate(c5, "failed")

    # 1. 精英池刷新测试
    manager.refresh_elite_pool(max_elite_size=2)
    
    # 2. 多样性采样测试 (在 active + elite 体系中优先剔除重复的 SrTiO3)
    diverse_samples = manager.sample_with_diversity(pool_name="active", sample_size=2)
    print(f"Diverse Samples (Should prefer distinct formulas): {[getattr(c.structure, 'formula') for c in diverse_samples]}")

    # 3. 统计接口输出
    stats = manager.get_statistics()
    print("Population Statistics Quick Look:")
    print(json.dumps(stats, indent=2))

    # 4. 索引查找
    print(f"Find by Formula 'SrTiO3' count: {len(manager.get_by_formula('SrTiO3'))}")
    print(f"Find by Generation 2 count: {len(manager.get_by_generation(2))}")

    # 5. 持久化存储与重载
    test_json_file = "population_snapshot.json"
    manager.save_to_json(test_json_file)
    
    # 建立一个新管理器尝试载入
    new_manager = PopulationManager()
    new_manager.load_from_json(test_json_file)
    print(f"Reloaded system active pool size: {len(new_manager.active_pool)}")
    
    # 清理测试文件
    import os
    if os.path.exists(test_json_file):
        os.remove(test_json_file)
