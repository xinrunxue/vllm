# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Task Pattern Analyzer for DynamicKV

Analyzes task patterns and provides layer-specific budget adjustments.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class TaskPattern:
    """任务模式定义
    
    Attributes:
        task_type: 任务类型标识
        layer_budget_ratios: 每层的预算调整比例
        typical_seq_length: 典型序列长度
        compression_tolerance: 压缩容忍度 (准确率下降容忍度)
        description: 任务描述
    """
    task_type: str
    layer_budget_ratios: list[float] = field(default_factory=list)
    typical_seq_length: int = 4096
    compression_tolerance: float = 0.1
    description: str = ""
    
    def get_adjusted_budget(self, base_budget: float, layer_idx: int) -> float:
        """获取调整后的层预算"""
        if layer_idx < len(self.layer_budget_ratios):
            return base_budget * self.layer_budget_ratios[layer_idx]
        return base_budget


class TaskPatternAnalyzer:
    """任务模式分析器
    
    职责：
    1. 检测当前任务类型
    2. 提供任务特定的预算调整建议
    3. 学习和适应新的任务模式
    
    基于论文观察：
    - RAG 任务：深层需要更多 token 来检索信息
    - 摘要任务：均匀压缩效果较好
    - Needle-in-a-Haystack：深层保留更多以找到关键信息
    - 代码任务：浅层保留更多（局部依赖性强）
    """
    
    TASK_PATTERNS: dict[str, TaskPattern] = {
        "rag": TaskPattern(
            task_type="rag",
            layer_budget_ratios=[
                1.2, 1.15, 1.1, 1.05, 1.0, 0.95, 0.9, 0.85, 0.8, 0.75,
                0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25,
            ],
            typical_seq_length=4096,
            compression_tolerance=0.1,
            description="Retrieval-Augmented Generation: deeper layers need more tokens for retrieval",
        ),
        "summarization": TaskPattern(
            task_type="summarization",
            layer_budget_ratios=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
            typical_seq_length=8192,
            compression_tolerance=0.15,
            description="Long text summarization: uniform compression works well",
        ),
        "needle_haystack": TaskPattern(
            task_type="needle_haystack",
            layer_budget_ratios=[
                0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15,
                1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65,
            ],
            typical_seq_length=16384,
            compression_tolerance=0.05,
            description="Needle-in-a-Haystack: deeper layers need more tokens to find key info",
        ),
        "code": TaskPattern(
            task_type="code",
            layer_budget_ratios=[
                1.5, 1.45, 1.4, 1.35, 1.3, 1.25, 1.2, 1.15, 1.1, 1.05,
                1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55,
            ],
            typical_seq_length=2048,
            compression_tolerance=0.1,
            description="Code generation: shallow layers need more tokens for local dependencies",
        ),
        "conversation": TaskPattern(
            task_type="conversation",
            layer_budget_ratios=[
                1.1, 1.1, 1.05, 1.05, 1.0, 1.0, 0.95, 0.95, 0.9, 0.9,
                0.85, 0.85, 0.8, 0.8, 0.75, 0.75, 0.7, 0.7, 0.65, 0.65,
            ],
            typical_seq_length=2048,
            compression_tolerance=0.12,
            description="Multi-turn conversation: moderate compression with recent token focus",
        ),
        "general": TaskPattern(
            task_type="general",
            layer_budget_ratios=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
            typical_seq_length=4096,
            compression_tolerance=0.1,
            description="General purpose: no task-specific adjustments",
        ),
    }
    
    def __init__(self, num_layers: int = 32):
        self.num_layers = num_layers
        self.current_task: Optional[str] = None
        self.task_history: list[str] = []
        self.confidence_scores: dict[str, float] = {}
        
        self._adjust_pattern_layers()
    
    def _adjust_pattern_layers(self):
        """调整预定义模式以匹配实际层数"""
        for task_type, pattern in self.TASK_PATTERNS.items():
            if len(pattern.layer_budget_ratios) != self.num_layers:
                original_ratios = pattern.layer_budget_ratios
                if len(original_ratios) > 0:
                    pattern.layer_budget_ratios = self._interpolate_ratios(
                        original_ratios, self.num_layers
                    )
                else:
                    pattern.layer_budget_ratios = [1.0] * self.num_layers
    
    def _interpolate_ratios(
        self, 
        original: list[float], 
        target_len: int
    ) -> list[float]:
        """插值生成目标长度的比例列表"""
        if len(original) == target_len:
            return original
        
        result = []
        for i in range(target_len):
            orig_idx = i * (len(original) - 1) / (target_len - 1)
            lower_idx = int(orig_idx)
            upper_idx = min(lower_idx + 1, len(original) - 1)
            frac = orig_idx - lower_idx
            
            interpolated = (
                original[lower_idx] * (1 - frac) + 
                original[upper_idx] * frac
            )
            result.append(interpolated)
        
        return result
    
    def detect_task_type(
        self,
        input_tokens: Optional[torch.Tensor] = None,
        prompt_text: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """自动检测任务类型
        
        Args:
            input_tokens: 输入 token 张量
            prompt_text: 原始提示文本
            metadata: 请求元数据
            
        Returns:
            检测到的任务类型
        """
        if metadata and "task_type" in metadata:
            return metadata["task_type"]
        
        if prompt_text:
            detected = self._detect_from_text(prompt_text)
            if detected != "general":
                return detected
        
        if input_tokens is not None:
            detected = self._detect_from_tokens(input_tokens)
            if detected != "general":
                return detected
        
        return "general"
    
    def _detect_from_text(self, text: str) -> str:
        """基于文本内容检测任务类型"""
        text_lower = text.lower()
        
        code_keywords = [
            "def ", "class ", "import ", "function", "return ",
            "```", "python", "javascript", "java", "c++",
        ]
        if any(kw in text_lower for kw in code_keywords):
            return "code"
        
        rag_keywords = [
            "context:", "document:", "passage:", "article:",
            "based on the following", "according to", "reference:",
        ]
        if any(kw in text_lower for kw in rag_keywords):
            return "rag"
        
        summary_keywords = [
            "summarize", "summary", "tldr", "brief overview",
            "main points", "key takeaways",
        ]
        if any(kw in text_lower for kw in summary_keywords):
            return "summarization"
        
        if "find" in text_lower or "locate" in text_lower or "search" in text_lower:
            if len(text) > 5000:
                return "needle_haystack"
        
        return "general"
    
    def _detect_from_tokens(self, tokens: torch.Tensor) -> str:
        """基于 token 特征检测任务类型"""
        seq_len = tokens.shape[-1] if tokens.dim() > 1 else len(tokens)
        
        if seq_len > 10000:
            return "needle_haystack"
        elif seq_len > 5000:
            return "summarization"
        
        return "general"
    
    def get_budget_adjustments(self, task_type: str) -> list[float]:
        """获取任务特定的预算调整因子
        
        Args:
            task_type: 任务类型
            
        Returns:
            每层的预算调整比例列表
        """
        if task_type in self.TASK_PATTERNS:
            pattern = self.TASK_PATTERNS[task_type]
            return pattern.layer_budget_ratios.copy()
        return [1.0] * self.num_layers
    
    def get_pattern(self, task_type: str) -> TaskPattern:
        """获取任务模式"""
        if task_type in self.TASK_PATTERNS:
            return self.TASK_PATTERNS[task_type]
        return self.TASK_PATTERNS["general"]
    
    def set_current_task(self, task_type: str):
        """设置当前任务类型"""
        if task_type not in self.TASK_PATTERNS:
            task_type = "general"
        self.current_task = task_type
        self.task_history.append(task_type)
        
        if len(self.task_history) > 100:
            self.task_history = self.task_history[-100:]
    
    def get_task_statistics(self) -> dict:
        """获取任务统计信息"""
        if not self.task_history:
            return {}
        
        stats = {}
        for task in self.task_history:
            stats[task] = stats.get(task, 0) + 1
        return stats
