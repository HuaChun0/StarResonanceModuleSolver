"""
模组搭配优化器
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from itertools import combinations
# from logging_config import get_logger # 假设您有此配置文件
from module_types import (
    ModuleInfo, ModuleType, ModuleAttrType, ModuleCategory,
    MODULE_CATEGORY_MAP, ATTR_THRESHOLDS
)

# 获取日志器
# logger = get_logger(__name__)
# 替换为基本日志记录以确保可独立运行
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModuleCombination:
    """模组搭配组合
    
    Attributes:
        modules: 模组列表
        total_attr_value: 总属性值 (根据阈值计算)
        attr_breakdown: 属性分布字典，键为属性名称，值为属性数值
        threshold_level: 达到的最高属性等级 (0-5)
        score: 综合评分（数值越大越好）
    """
    modules: List[ModuleInfo]
    total_attr_value: int
    attr_breakdown: Dict[str, int]  # 属性名称 -> 原始总值
    threshold_level: int  # 组合中出现的最高属性等级
    score: float  # 综合评分


class ModuleOptimizer:
    """模组搭配优化器"""
    
    def __init__(self):
        self.logger = logger
    
    def get_module_category(self, module: ModuleInfo) -> ModuleCategory:
        """获取模组类型分类"""
        return MODULE_CATEGORY_MAP.get(module.config_id, ModuleCategory.ATTACK)
    
    def calculate_total_attr_value(self, modules: List[ModuleInfo]) -> Tuple[int, Dict[str, int]]:
        """计算模组组合的总属性值 (此方法用于计算用于显示的阈值化总值)"""
        attr_breakdown = {}
        for module in modules:
            for part in module.parts:
                attr_name = part.name
                attr_breakdown[attr_name] = attr_breakdown.get(attr_name, 0) + part.value
        
        total_threshold_value = 0
        for attr_name, attr_value in attr_breakdown.items():
            threshold_value = 0
            for threshold in ATTR_THRESHOLDS:
                if attr_value >= threshold:
                    threshold_value = threshold
                else:
                    break
            total_threshold_value += threshold_value
        
        return total_threshold_value, attr_breakdown

    def calculate_combination_score_v2(self, attr_breakdown: Dict[str, int]) -> float:
        """
        计算组合的综合评分（V2新算法）。
        - 强烈奖励达到高阈值 (16, 20) 的属性。
        - 惩罚属性值超过20的浪费。
        - 鼓励多个属性达到高等级。
        """
        total_score = 0.0
        
        # 阈值及其对应的基础分，指数级增长以突出高等级的重要性
        tier_scores = {20: 100000, 16: 50000, 12: 15000, 8: 5000, 4: 1000, 1: 100}
        
        num_high_tier_attrs = 0

        for attr_name, attr_value in attr_breakdown.items():
            # 1. 计算浪费值 (超过20的部分)
            waste = max(0, attr_value - 20)
            
            # 2. 找到达到的最高阈值
            achieved_threshold = 0
            for t in sorted(ATTR_THRESHOLDS, reverse=True):
                if attr_value >= t:
                    achieved_threshold = t
                    break
            
            # 3. 计算该属性的得分
            attr_score = 0.0
            if achieved_threshold > 0:
                # 基础分
                attr_score += tier_scores.get(achieved_threshold, 0)
                
                # 加上原始属性值作为次要分数，用于区分同阈值下的不同数值 (例如17分优于16分)
                attr_score += attr_value * 10
                
                # 惩罚浪费值，每浪费1点，惩罚力度巨大
                waste_penalty = waste * 2500
                attr_score -= waste_penalty

            total_score += attr_score

            # 统计达到高等级(>=16)的词条数量
            if achieved_threshold >= 16:
                num_high_tier_attrs += 1
        
        # 4. 多高等级属性奖励: 每增加一个高等级词条，都给予额外加分
        if num_high_tier_attrs > 1:
            total_score += (num_high_tier_attrs - 1) * 75000
            
        return total_score
   
    def find_optimal_combinations(self, modules: List[ModuleInfo], category: ModuleCategory, top_n: int = 20) -> List[ModuleCombination]:
        """
        [修正后] 通过启发式精英池搜索，寻找最优模组搭配。
        此算法优先考虑那些在特定属性上表现突出的模组，以更大概率找到能凑出高等级词条的组合。
        
        Args:
            modules: 所有模组列表
            category: 目标模组类型（攻击/守护/辅助）
            top_n: 返回前N个最优组合, 默认20
            
        Returns:
            最优模组组合列表，按新评分系统排序
        """
        self.logger.info(f"开始计算 {category.value} 类型模组的最优搭配 (启发式精英池算法)")
        
        # 1. 按类型过滤模组
        filtered_modules = [m for m in modules if self.get_module_category(m) == category]
        self.logger.info(f"找到 {len(filtered_modules)} 个 {category.value} 类型模组")
        
        if len(filtered_modules) < 4:
            self.logger.warning(f"{category.value} 类型模组数量不足4个, 无法形成搭配")
            return []
        
        # 2. 按每个独立属性，对模组进行排序，构建“单科状元”列表
        attr_sorted_modules: Dict[str, List[ModuleInfo]] = {}
        all_attrs = {part.name for module in filtered_modules for part in module.parts}

        for attr_name in all_attrs:
            modules_with_attr = []
            for module in filtered_modules:
                attr_value = next((part.value for part in module.parts if part.name == attr_name), 0)
                if attr_value > 0:
                    modules_with_attr.append((module, attr_value))
            modules_with_attr.sort(key=lambda x: x[1], reverse=True)
            attr_sorted_modules[attr_name] = [m[0] for m in modules_with_attr]

        # 3. 创建精英候选池 (使用字典替代集合来去重)
        elite_pool_dict: Dict[str, ModuleInfo] = {}
        
        # 从每个属性的排序列表顶部挑选模组加入池中，确保“专才”不被埋没
        num_to_pick_per_attr = 10
        for sorted_list in attr_sorted_modules.values():
            for module in sorted_list[:num_to_pick_per_attr]:
                elite_pool_dict[module.uuid] = module
        
        # 如果精英池数量过少，用总属性值高的模组补充，保证组合多样性
        if len(elite_pool_dict) < 20 and len(filtered_modules) > len(elite_pool_dict):
            sorted_by_total = sorted(filtered_modules, key=lambda m: sum(p.value for p in m.parts), reverse=True)
            for module in sorted_by_total:
                if module.uuid not in elite_pool_dict:
                    elite_pool_dict[module.uuid] = module
                if len(elite_pool_dict) >= 20:
                    break
        
        candidate_modules = list(elite_pool_dict.values())

        self.logger.info(f"创建了 {len(candidate_modules)} 个模组的精英候选池进行组合计算")
        
        # 4. 从精英池生成组合 (为防止性能问题，可对候选池大小进行限制)
        if len(candidate_modules) > 40:
            self.logger.warning(f"精英池过大({len(candidate_modules)})，将截取总属性值最高的前40个")
            candidate_modules.sort(key=lambda m: sum(p.value for p in m.parts), reverse=True)
            candidate_modules = candidate_modules[:40]

        combinations_list = list(combinations(candidate_modules, 4))
        self.logger.info(f"从精英池生成了 {len(combinations_list)} 个4模组组合")
        
        # 5. 使用新评分系统评估所有组合
        module_combinations = []
        for combo_modules in combinations_list:
            # 计算原始属性分布
            _, attr_breakdown = self.calculate_total_attr_value(list(combo_modules))
            
            # 使用新评分函数计算分数
            score = self.calculate_combination_score_v2(attr_breakdown)
            
            # 计算用于显示的阈值化总值
            total_threshold_value, _ = self.calculate_total_attr_value(list(combo_modules))
            
            # 计算组合达到的最高属性等级
            highest_threshold_level = -1 # 初始化为-1表示无任何等级
            for value in attr_breakdown.values():
                level = -1
                for i, threshold in enumerate(ATTR_THRESHOLDS):
                    if value >= threshold:
                        level = i
                if level > highest_threshold_level:
                    highest_threshold_level = level

            combination = ModuleCombination(
                modules=list(combo_modules),
                total_attr_value=total_threshold_value,
                attr_breakdown=attr_breakdown,
                threshold_level=highest_threshold_level,
                score=score
            )
            module_combinations.append(combination)
        
        # 6. 按新评分降序排序，返回最佳结果
        module_combinations.sort(key=lambda x: x.score, reverse=True)
        return module_combinations[:top_n]

    def print_combination_details(self, combination: ModuleCombination, rank: int):
        """打印组合详细信息"""
        print(f"\n=== 第{rank}名搭配 ===")
        # 达到属性等级的描述现在基于最高的单个属性
        level_desc = f"{ATTR_THRESHOLDS[combination.threshold_level]}点" if combination.threshold_level >= 0 else "无"
        print(f"最高属性等级: {combination.threshold_level} ({level_desc})")
        print(f"综合评分: {combination.score:.1f}")
        
        print("\n模组列表:")
        for i, module in enumerate(combination.modules, 1):
            parts_str = ", ".join([f"{p.name}+{p.value}" for p in module.parts])
            # [修正] 将 uuid 强制转换为字符串再切片，防止其为整数时报错
            uuid_str = str(module.uuid)
            print(f"  {i}. {module.name} (品质{module.quality}, UUID:{uuid_str[:6]}) - {parts_str}")
        
        print("\n属性分布 (原始总值):")
        for attr_name, value in sorted(combination.attr_breakdown.items()):
            print(f"  {attr_name}: +{value}")
    
    def optimize_and_display(self, 
                           modules: List[ModuleInfo], 
                           category: ModuleCategory = ModuleCategory.ATTACK,
                           top_n: int = 20):
        """优化并显示结果"""
        print(f"\n{'='*50}")
        print(f"模组搭配优化 - {category.value}类型")
        print(f"{'='*50}")
        
        # 调用重构后的主函数
        optimal_combinations = self.find_optimal_combinations(modules, category, top_n)
        
        if not optimal_combinations:
            print(f"未找到{category.value}类型的有效搭配")
            return
        
        print(f"\n找到{len(optimal_combinations)}个最优搭配:")
        
        for i, combination in enumerate(optimal_combinations, 1):
            self.print_combination_details(combination, i)
        
        print(f"\n{'='*50}")
        print("统计信息:")
        print(f"总模组数量: {len(modules)}")
        print(f"{category.value}类型模组: {len([m for m in modules if self.get_module_category(m) == category])}")
        if optimal_combinations:
            print(f"最高分搭配评分: {optimal_combinations[0].score:.1f}")
            print(f"最高分搭配的最高属性等级: {optimal_combinations[0].threshold_level}")
        print(f"{'='*50}")
