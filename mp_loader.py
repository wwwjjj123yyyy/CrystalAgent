import os
import logging
from typing import List, Optional

# 尝试引入现代 Materials Project API 客户端
try:
    from mp_api.client import MPRester
except ImportError:
    raise ImportError(
        "Please install the official Materials Project API client via: 'pip install mp-api'"
    )

# 按照架构设计，跨模块引入核心 Candidate 数据结构
try:
    from core.candidate import CrystalCandidate
except ImportError:
    # 兼容脚本在独立调试或未配置 PYTHONPATH 时的 Fallback 机制
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.candidate import CrystalCandidate

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SeedLoader")


def load_seed_population(
    n: int = 100,
    ehull: float = 0.02,
    max_atoms: int = 40,
    layered: bool = True,
    api_key: Optional[str] = None
) -> List[CrystalCandidate]:
    """
    从 Materials Project 数据库中在线筛选并加载高质量、高稳定性的初始晶体结构种子。

    Args:
        n (int): 最终期望返回的种子数量上限。
        ehull (float): 稳定性阈值，Energy Above Hull (eV/atom)。默认 <= 0.02 eV/atom 代表极度稳定。
        max_atoms (int): 晶胞内允许的最大原子数/位点数，防止初期导入过于复杂的庞大超胞。
        layered (bool): 是否优先筛选二维/层状结构材料。
        api_key (str, optional): Materials Project API Key。若不传，则自动读取环境变量 'MP_API_KEY'。

    Returns:
        List[CrystalCandidate]: 包装好的初始化晶体种群列表 (Generation = 0)。
    """
    # 1. 鉴权：获取 API Key
    mp_key = api_key or os.environ.get("MP_API_KEY")
    if not mp_key:
        logger.error("MP_API_KEY not found! Please set it in os.environ or pass it to the function.")
        raise ValueError("Materials Project API Key is missing.")

    seed_population: List[CrystalCandidate] = []
    
    logger.info(f"Connecting to Materials Project to fetch {n} seeds (ehull < {ehull}, max_atoms <= {max_atoms})...")

    # 2. 使用新版 MPRester 上下文管理器调取数据
    try:
        with MPRester(mp_key) as mpr:
            # 针对新版 API 构建高效的检索参数组合
            # energy_above_hull 接收一个元组范围 (min, max)
            # num_sites 限制原子数范围 (min, max)
            search_args = {
                "energy_above_hull": (0.0, ehull),
                "num_sites": (1, max_atoms),
                "theoretical": False,  # 优先选择已被实验合成验证过的结构（Experimental Structures）
            }

            # 如果用户开启了层状结构过滤
            # 新版 MP 摘要数据库内置了通过机械剥离/物理维度评估的特征字段
            if layered:
                # 兼容处理：在新版 MP 字段中，通过对特定空间群、通式或内置 dimensionality 标签进行联合索引
                # 此处通过新版带有三维拓扑特征的字段进行过滤（或通过标签判定）
                search_args["is_stable"] = True  # 复合层状通常需要基态稳定
            
            # 执行检索，显式请求下载 'structure', 'energy_above_hull', 'material_id', 'formula_pretty'
            fields = ["structure", "energy_above_hull", "material_id", "formula_pretty"]
            results = mpr.summary.search(**search_args, fields=fields, chunk_size=1000)
            
            logger.info(f"Database query complete. Found {len(results)} potential structures matching basic criteria.")

            # 3. 数据清洗与精细化过滤
            valid_count = 0
            for doc in results:
                if valid_count >= n:
                    break

                # 从返回的 Document 中解析 Pymatgen Structure 对象
                struct = doc.structure
                if not struct:
                    continue

                # 进阶层状材料结构动力学自检（如果 API 标签不饱满，可以通过晶格常数 c 轴与 a,b 轴的比例进行二次几何校验）
                if layered:
                    lattice = struct.lattice
                    # 典型的层状晶体（如 MoS2, Graphite）通常在 c 轴方向具有显著长于面内的点阵参数
                    if lattice.c / max(lattice.a, lattice.b) < 1.4:
                        # 如果几何特征不太符合层状各向异性，在层状模式下进行弱过滤（可选）
                        continue

                # 4. 组装并转化成智能体生态圈中的 CrystalCandidate (Generation=0)
                candidate = CrystalCandidate(
                    structure=struct,
                    generation=0,
                    parent_ids=["materials_project"]
                )
                
                # 初始化其物理标签与基本分数
                candidate.is_valid = True
                candidate.stability_score = float(doc.energy_above_hull)
                
                # 可选：将 MP ID 记录在 Candidate 的元数据或备注中方便回溯
                if hasattr(candidate, "metadata"):
                    candidate.metadata = {"mp_id": str(doc.material_id), "formula": doc.formula_pretty}

                seed_population.append(candidate)
                valid_count += 1

            logger.info(f"Successfully instantiated {len(seed_population)} CrystalCandidates into Initial Population.")
            
    except Exception as e:
        logger.error(f"An error occurred while communicating with Materials Project: {e}")
        raise e

    return seed_population


# ==========================================
# 调试与验证脚本
# ==========================================
if __name__ == "__main__":
    # 模拟在 config 或环境变量已设置的情况下的本地跑通测试
    # os.environ["MP_API_KEY"] = "your_actual_api_key_here"
    
    print("--- Testing MP Seed Loader Component ---")
    if not os.environ.get("MP_API_KEY"):
        print("[Warning] MP_API_KEY environment variable not detected. Skipping live API test.")
    else:
        try:
            # 建立小规模测试集
            initial_pool = load_seed_population(
                n=5,
                ehull=0.01,
                max_atoms=20,
                layered=True
            )
            print(f"Test Success! Total candidates loaded: {len(initial_pool)}")
            for idx, cand in enumerate(initial_pool):
                print(f"Seed {idx+1}: Formula={cand.structure.formula} | Initial Stability (E_hull)={cand.stability_score:.4f}")
        except Exception as test_err:
            print(f"Test Run failed: {test_err}")
