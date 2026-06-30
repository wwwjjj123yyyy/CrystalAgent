import os
import json
import logging
from typing import Dict, Any, Optional

# DeepSeek 官方推荐使用标准 OpenAI SDK 进行接入兼容
try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "Please install the official OpenAI SDK to connect with DeepSeek via: 'pip install openai'"
    )

# 配置日志
logger = logging.getLogger("DeepSeekClient")
logger.setLevel(logging.INFO)


class DeepSeekClient:
    """
    DeepSeek 大模型客户端包装器
    专门处理 Prompt -> API -> Clean JSON 的高可靠数据流
    """
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-pro"  # 选用高性能生产模型，若需快速低成本演化可换为 "deepseek-v4-flash"
    ):
        """
        Args:
            api_key: DeepSeek API 密钥，若为 None 则自动读取环境变量 DEEPSEEK_API_KEY
            base_url: DeepSeek 官方 API 端点
            model: 调用的模型名称
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            logger.error("DEEPSEEK_API_KEY not found in environment or arguments.")
            raise ValueError("Missing DeepSeek API Key.")
            
        self.model = model
        
        # 初始化兼容的 OpenAI 客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )

    def request_structured_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> Dict[str, Any]:
        """
        向 DeepSeek 发起请求，并通过 JSON Mode 强制返回结构化 JSON。
        
        Args:
            system_prompt: 角色与返回 JSON Schema 约束定义
            user_prompt: 当前代次的进化统计数据与父体特征
            temperature: 采样温度（低温度如 0.3 有利于生成符合格式要求的稳定架构）
            
        Returns:
            Dict[str, Any]: 解析成功后的 Python 字典
        """
        # 注意：DeepSeek 启用 JSON Mode 的核心前置条件：
        # 1. 必须设置 response_format={"type": "json_object"}
        # 2. 提示词（System 或 User）中必须包含 "json" 这个单词/词汇进行显式引导
        
        try:
            logger.info(f"Sending workflow request to DeepSeek ({self.model})...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=2048  # 给足 Token 防止 JSON 骨架因截断导致失效
            )
            
            raw_content = response.choices[0].message.content
            if not raw_content:
                raise ValueError("DeepSeek returned an empty response content.")
                
            # [工程冷知识] 检查是否有推理链（CoT）内容，如果有可以记录下来用于调试反思
            if hasattr(response.choices[0].message, "reasoning_content"):
                cot_content = response.choices[0].message.reasoning_content
                if cot_content:
                    logger.debug(f"DeepSeek CoT Process:\n{cot_content}")
            
            # 将拿到的字符串直接反序列化为 JSON 字典
            parsed_json = json.loads(raw_content)
            return parsed_json

        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse malformed JSON from DeepSeek response: {json_err}")
            logger.error(f"Raw problematic output: {raw_content}")
            # 熔断治理：返回空字典或触发重试机制，防止 run.py 主流中断
            return {}
            
        except Exception as e:
            logger.error(f"An error occurred during DeepSeek API communication: {e}")
            return {}


# ==========================================
# 单元测试与管道集成演示
# ==========================================
if __name__ == "__main__":
    print("--- Testing DeepSeekClient Component ---")
    
    # 构造模拟的 System 角色与格式范例 (prompts.py 的微缩版)
    mock_system = """
    You are the Crystal Evolution Planner. Analyze the past performance and suggest the next mutation step.
    You must output your decision in strict JSON format.
    Example:
    {
        "thought": "Replace species performs best, I will continue with it.",
        "next_mutation_strategy": "replace_species",
        "mutation_strength": "MEDIUM"
    }
    """
    
    # 构造模拟的上一代状态数据 (mutation_statistics)
    mock_user = """
    Current generation stats report:
    {
        "replace_species": {"success": 18, "failed": 2, "average_fitness": 0.64},
        "layer_shift": {"success": 1, "failed": 14, "average_fitness": 0.12}
    }
    Please output the JSON for next generation step.
    """
    
    # 本地跑通检测（需配置环境变量）
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("[Warning] DEEPSEEK_API_KEY not found in env. Skipping live integration test.")
    else:
        client = DeepSeekClient()
        json_res = client.request_structured_json(mock_system, mock_user)
        print("\n=== Successfully Parsed JSON From DeepSeek API ===")
        print(type(json_res))
        print(json.dumps(json_res, indent=4, ensure_ascii=False))
