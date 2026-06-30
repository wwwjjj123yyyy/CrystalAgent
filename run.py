import os
import json
import logging
import random
from typing import List, Dict, Any

# ==========================================
# 1. 跨模块引入系统组件
# ==========================================
from seed.mp_loader import load_seed_population
from evolution.mutation import MutationEngine, MutationType, MutationStrength
from physics.validator import CrystalValidator
from evaluator.evaluator_wrapper import EvaluatorWrapper
from llm.deepseek_client import DeepSeekClient

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(name)s: %(message)s')
logger = logging.getLogger("EvolutionOrchestrator")

# 进化控制超参数
MAX_GENERATIONS = 5     # 演化代数
POPULATION_SIZE = 20     # 每代种群的维持规模
OFFSPRING_SIZE = 15      # 每代产生的子代数量


# ==========================================
# [架构补充] 快速对齐 core/population.py 的管理组件
# ==========================================
class PopulationManager:
    """负责精细化管理当代存活者(active)、淘汰者(failed)以及历史最强精英(elites)"""
    def __init__(self):
        self.active: List[Any] = []
        self.failed: List[Any] = []
        self.elites: List[Any] = []

    def add_candidate(self, candidate, pool_name: str = "active"):
        if pool_name == "active":
            self.active.append(candidate)
            # 动态维护全局最高分精英
            self.elites.append(candidate)
            self.elites = sorted(self.elites, key=lambda x: x.fitness, reverse=True)[:5]
        elif pool_name == "failed":
            self.failed.append(candidate)

    def get_summary(self) -> Dict[str, Any]:
        """为 Planner 大模型提供高价值的种群快照"""
        if not self.active:
            return {"status": "Empty population"}
        best = max(self.active, key=lambda x: x.fitness)
        avg_fitness = sum(x.fitness for x in self.active) / len(self.active)
        return {
            "total_active_count": len(self.active),
            "total_failed_count": len(self.failed),
            "best_fitness_ever": best.fitness,
            "best_formula": getattr(best.structure, "formula", "Unknown"),
            "average_fitness_current": round(avg_fitness, 4)
        }


# ==========================================
# [架构补充] 快速对齐 evolution/selection.py 的选择机制
# ==========================================
class SelectionMechanism:
    """遗传选择器：采用轮盘赌或锦标赛算法挑选优质父母"""
    @staticmethod
    def tournament_select(population: List[Any], k: int = 3) -> Any:
        """锦标赛选择：随机挑k个，谁得分高谁当父母"""
        aspirants = random.sample(population, min(k, len(population)))
        return max(aspirants, key=lambda x: x.fitness)


# ==========================================
# 2. 演化主工作流引擎
# ==========================================
def run_evolution():
    logger.info("==============================================")
    logger.info("Initializing Crystal Evolution Agent Engine...")
    logger.info("==============================================")

    # 2.1 实例化各核心组件
    pop_manager = PopulationManager()
    mutation_engine = MutationEngine()
    validator = CrystalValidator()
    evaluator = EvaluatorWrapper(w_energy=0.6, w_comp_novelty=0.2, w_struct_novelty=0.2)
    
    # 鉴权检查，若无实际API Key则降级为随机Planner
    has_llm = "DEEPSEEK_API_KEY" in os.environ
    llm_planner = DeepSeekClient() if has_llm else None
    if not has_llm:
        logger.warning("DEEPSEEK_API_KEY not found. Running in Random Planner fallback mode.")

    # ==========================================
    # 步骤一：Population (初始化种群种子)
    # ==========================================
    logger.info("\n>>> Step 1: Loading Initial Seed Population from Materials Project...")
    try:
        # 在线获取 20 个高稳定性层状材料种子
        seeds = load_seed_population(n=POPULATION_SIZE, ehull=0.02, max_atoms=30, layered=True)
    except Exception:
        logger.warning("Failed to fetch seeds online. Using Mock Seed Data for pipeline testing.")
        # 本地 Mock 兜底种子
        from evolution.mutation import Structure, CrystalCandidate
        seeds = [
            CrystalCandidate(Structure(None, ["Bi", "Te"], [[0,0,0], [0.5,0.5,0.5]]), generation=0)
            for _ in range(POPULATION_SIZE)
        ]
    
    # 初始化种子的评估得分并归档
    for seed in seeds:
        seed.is_valid = True
        evaluator.process(seed)
        pop_manager.add_candidate(seed, pool_name="active")

    # ==========================================
    # 主演化循环：Repeat
    # ==========================================
    for gen in range(1, MAX_GENERATIONS + 1):
        logger.info(f"\n==============================================")
        logger.info(f"🧬 STARTING GENERATION {gen} / {MAX_GENERATIONS}")
        logger.info(f"==============================================")

        # 提取当前系统的运行状态快照
        pop_summary = pop_manager.get_summary()
        mutation_stats = mutation_engine.statistics.get_report()
        
        logger.info(f"Current Status: Best Fitness={pop_summary.get('best_fitness_ever')}, Avg Fitness={pop_summary.get('average_fitness_current')}")

        # 大模型大脑反思决策层 (LLM Planner Decision)
        chosen_strategy = {}
        if llm_planner:
            system_prompt = (
                "You are the Lead Crystal Evolution Planner. Analyze the mutation statistics and population status, "
                "then output a JSON deciding the next strategic action plan. Your JSON must contain exactly these fields:\n"
                "- 'reasoning': text analyzing which operators perform well/poorly.\n"
                "- 'recommended_mutation_type': string representing the best MutationType value.\n"
                "- 'recommended_strength': string representing MutationStrength value."
            )
            user_prompt = f"Population Status:\n{json.dumps(pop_summary, indent=2)}\n\nMutation Operator Stats:\n{json.dumps(mutation_stats, indent=2)}"
            chosen_strategy = llm_planner.request_structured_json(system_prompt, user_prompt)
            logger.info(f"LLM Planner Insight: {chosen_strategy.get('reasoning')}")
        
        # 解析进化策略（若无LLM则随机选取）
        next_op_val = chosen_strategy.get("recommended_mutation_type", random.choice([m.value for m in MutationType]))
        next_str_val = chosen_strategy.get("recommended_strength", "MEDIUM")
        
        planned_op = MutationType(next_op_val)
        planned_strength = MutationStrength(next_str_val)
        
        logger.info(f"🧬 [Strategy Selected] Planner deployed operator: [{planned_op.value}] with [{planned_strength.value}] strength.")

        # 开始批量繁衍子代
        offspring_count = 0
        while offspring_count < OFFSPRING_SIZE:
            
            # ==========================================
            # 步骤二：Selection (挑选优质父母)
            # ==========================================
            parent = SelectionMechanism.tournament_select(pop_manager.active, k=3)
            
            # ==========================================
            # 步骤三：Mutation (算子突变)
            # ==========================================
            # 深度拷贝父结构并送往引擎变异
            child_structure = mutation_engine.apply(parent.structure, planned_op, planned_strength)
            
            # 包装为新一代 Candidate
            from evolution.mutation import CrystalCandidate
            child = CrystalCandidate(structure=child_structure, generation=gen, parent_ids=[parent.id])

            # ==========================================
            # 步骤四：Validator (物理硬约束判定与分流)
            # ==========================================
            is_legit, error_msg = validator.validate(child)
            
            if not is_legit:
                # 触发硬熔断：直接归档 failed 池，向引擎注册失败，免除后续计算
                child.is_valid = False
                child.error_msg = error_msg
                pop_manager.add_candidate(child, pool_name="failed")
                mutation_engine.statistics.record(planned_op, is_success=False)
                continue  # 继续繁衍下一个
                
            # ==========================================
            # 步骤五：Evaluator (多指标深度评估)
            # ==========================================
            child.is_valid = True
            evaluator.process(child)
            
            # 登记成功晋级个体，回填收益至突变历史数据看板
            pop_manager.add_candidate(child, pool_name="active")
            mutation_engine.statistics.record(planned_op, is_success=True, fitness=child.fitness)
            offspring_count += 1

        # ==========================================
        # 步骤六：Selection (环境大淘沙：精简重组种群)
        # ==========================================
        # 每一代结束时，只保留全生态圈里表现最优秀的 POPULATION_SIZE 个体进入下一代轮转
        all_survivors = sorted(pop_manager.active, key=lambda x: x.fitness, reverse=True)
        pop_manager.active = all_survivors[:POPULATION_SIZE]
        
        logger.info(f"Generation {gen} complete. Active survivors retained: {len(pop_manager.active)}. Dead pool size: {len(pop_manager.failed)}")

    # 演化全程结束，汇报终极大奖
    logger.info("\n==============================================")
    logger.info("🎉 EVOLUTIONARY SEARCH COMPLETE 🎉")
    logger.info("==============================================")
    best_crystal = pop_manager.elites[0]
    logger.info(f"Top 1 Elite Crystal ID: {best_crystal.id}")
    logger.info(f"Max Fitness Achieved: {best_crystal.fitness}")
    logger.info(f"Detailed Sub-Metrics: {getattr(best_crystal, 'metrics', {})}")
    logger.info(f"Structure Discovery Formula: {getattr(best_crystal.structure, 'formula', 'Unknown')}")


if __name__ == "__main__":
    # 执行启动开关
    run_evolution()
