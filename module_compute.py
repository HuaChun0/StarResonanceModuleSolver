# module_optimizer_calculations.py
from typing import List, Tuple, Dict, Any, Optional
from itertools import combinations
import random
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from math import comb
import multiprocessing
import heapq
import gc
import itertools

_DEF_THRESHOLDS = [20, 40, 60, 80, 100]  # 保守默认
_DEF_ATTR_TYPE_MAP: Dict[str, str] = {}  # 默认为 basic
_DEF_BASIC_POWER = {1: 1, 2: 3, 3: 7, 4: 12, 5: 20, 6: 30}
_DEF_SPEC_POWER  = {1: 1, 2: 3, 3: 7, 4: 12, 5: 20, 6: 30}
# 总属性值战力的默认映射：按总值的粗略阶梯
_DEF_TOTAL_ATTR_POWER = {v: int(v * 0.1) for v in range(0, 10001, 10)}

def _get_global_or_default(name: str, default):
    return globals().get(name, default)

try:
    from module_types import (
        ATTR_THRESHOLDS as _EXT_THRESHOLDS,
        ATTR_NAME_TYPE_MAP as _EXT_ATTR_TYPE_MAP,
        BASIC_ATTR_POWER_MAP as _EXT_BASIC_POWER,
        SPECIAL_ATTR_POWER_MAP as _EXT_SPEC_POWER,
        TOTAL_ATTR_POWER_MAP as _EXT_TOTAL_POWER,
    )
    # 注入到本模块全局，供 _get_global_or_default() 读取
    ATTR_THRESHOLDS = _EXT_THRESHOLDS
    ATTR_NAME_TYPE_MAP = _EXT_ATTR_TYPE_MAP
    BASIC_ATTR_POWER_MAP = _EXT_BASIC_POWER
    SPECIAL_ATTR_POWER_MAP = _EXT_SPEC_POWER
    TOTAL_ATTR_POWER_MAP = _EXT_TOTAL_POWER
except Exception:
    # 没有 module_types 时，继续使用本文件的默认表
    pass


# 数据结构
class ModuleSolution:
    """
    模块搭配解
    Attributes:
        modules: 模块列表（UI 字典）
        score:   综合评分（现在改为战斗力）
        attr_breakdown: 属性分布 {attr_name: total_points}
    """
    def __init__(self, modules: List[Dict[str, Any]], score: float, attr_breakdown: Dict[str, int]):
        self.modules = modules
        self.score = score
        self.attr_breakdown = attr_breakdown


# 主计算器
class ModuleOptimizerCalculations:
    """
    计算模块（支持 UI 字典结构：{"a1","p1","a2","p2","a3","p3"}）
    功能：
      - 属性桶（跳过被排斥属性；先过滤掉含排斥属性的模块；每桶 Top-N=30）
      - 贪心 + 局部搜索（目标：战斗力）
      - 枚举（仅对排除后剩余模块；评分：战斗力）
    """

    def __init__(
        self,
        exclude_set: Optional[set] = None,
        thresholds: Optional[List[int]] = None,
        attr_type_map: Optional[Dict[str, str]] = None,
        basic_attr_power_map: Optional[Dict[int, int]] = None,
        special_attr_power_map: Optional[Dict[int, int]] = None,
        total_attr_power_map: Optional[Dict[int, int]] = None,
        bucket_cap: int = 30,
        slots: int = 4,
    ):
        self.exclude_set = exclude_set or set()
        self.thresholds = thresholds or _get_global_or_default("ATTR_THRESHOLDS", _DEF_THRESHOLDS)
        self.attr_type_map = attr_type_map or _get_global_or_default("ATTR_NAME_TYPE_MAP", _DEF_ATTR_TYPE_MAP)
        self.basic_attr_power = basic_attr_power_map or _get_global_or_default("BASIC_ATTR_POWER_MAP", _DEF_BASIC_POWER)
        self.special_attr_power = special_attr_power_map or _get_global_or_default("SPECIAL_ATTR_POWER_MAP", _DEF_SPEC_POWER)
        self.total_attr_power = total_attr_power_map or _get_global_or_default("TOTAL_ATTR_POWER_MAP", _DEF_TOTAL_ATTR_POWER)
        self.bucket_cap = int(bucket_cap)
        self.slots = int(slots)
        self._tie = itertools.count()

    def _get_mod_uuid(self, m: Any) -> str:
        """
        通用获取模块唯一ID：
        - UI 字典：优先取 m['uuid'/'uid'/'id']；否则用 (a1,p1,a2,p2,a3,p3) 生成签名
        - ModuleInfo：优先取 .uuid / .uid / .id；否则用 .parts 的 (name,value) 组合生成签名
        """
        # UI dict
        if isinstance(m, dict):
            for k in ("uuid", "uid", "id"):
                if k in m and m[k]:
                    return str(m[k])
            a1, p1 = str(m.get("a1", "")).strip(), int(m.get("p1", 0))
            a2, p2 = str(m.get("a2", "")).strip(), int(m.get("p2", 0))
            a3, p3 = str(m.get("a3", "")).strip(), int(m.get("p3", 0))
            return f"UD|{a1}|{p1}|{a2}|{p2}|{a3}|{p3}"

        # ModuleInfo-like
        for k in ("uuid", "uid", "id"):
            v = getattr(m, k, None)
            if v not in (None, ""):
                return str(v)
        parts = getattr(m, "parts", [])
        sig = "|".join(f"{getattr(p, 'name', '')}:{int(getattr(p, 'value', 0))}" for p in parts)
        return f"MI|{sig}"

    # UI 模块工具
    def _ui_iter_attr_pairs(self, mod_dict: Dict[str, Any]):
        """从 UI 模块字典取 (attr, points)。自动跳过不存在的第三词条。"""
        a1, p1 = mod_dict.get("a1"), int(mod_dict.get("p1", 0))
        a2, p2 = mod_dict.get("a2"), int(mod_dict.get("p2", 0))
        if a1 is not None:
            yield a1, p1
        if a2 is not None:
            yield a2, p2
        a3 = mod_dict.get("a3", None)
        if a3 is not None:
            yield a3, int(mod_dict.get("p3", 0))

    def _ui_mod_has_excluded_attr(self, mod_dict: Dict[str, Any]) -> bool:
        """该模块是否包含被排斥属性。"""
        if not self.exclude_set:
            return False
        for name, _ in self._ui_iter_attr_pairs(mod_dict):
            if name in self.exclude_set:
                return True
        return False

    def _ui_combo_to_pts(self, mods: List[Dict[str, Any]]) -> Dict[str, int]:
        """组合 -> 属性分布聚合。"""
        pts: Dict[str, int] = {}
        for g in mods:
            for name, val in self._ui_iter_attr_pairs(g):
                pts[name] = pts.get(name, 0) + int(val)
        return pts

    def combat_power_from_attr_breakdown(self, attr_breakdown: Dict[str, int]) -> int:
        """
        战斗力计算（与 optimizer 一致）：阈值战力 + 总属性战力
        阈值战力按属性类型 basic/special 分别查表；总属性战力按总值查表。
        """
        threshold_power = 0
        for attr_name, raw_val in attr_breakdown.items():
            v = int(raw_val)

            # 阈值等级
            level = 0
            for i, t in enumerate(self.thresholds):
                if v >= t:
                    level = i + 1
                else:
                    break
            if level > 0:
                attr_type = self.attr_type_map.get(attr_name, "basic")
                pmap = self.special_attr_power if attr_type == "special" else self.basic_attr_power
                threshold_power += int(pmap.get(level, 0))

        total_attr_value = sum(int(v) for v in attr_breakdown.values())
        total_attr_power = int(self.total_attr_power.get(total_attr_value, 0))
        return int(threshold_power + total_attr_power)

    def high_level_score_from_attr_breakdown(self, attr_breakdown: Dict[str, int]) -> float:
        """
        高等级优先评分：重点关注5级和6级属性的数量
        """
        high_level_count = 0
        level_bonus = 0.0
        
        for attr_name, raw_val in attr_breakdown.items():
            v = int(raw_val)
            
            # 计算阈值等级
            level = 0
            for i, t in enumerate(self.thresholds):
                if v >= t:
                    level = i + 1
                else:
                    break
            
            # 高等级奖励：5级和6级给予额外加分
            if level >= 5:
                high_level_count += 1
                level_bonus += (level - 4) * 100  # 5级+100，6级+200
            elif level >= 3:
                level_bonus += level * 10  # 3级+30，4级+40
            elif level > 0:
                level_bonus += level * 5   # 1级+5，2级+10
        
        # 基础战斗力
        base_power = self.combat_power_from_attr_breakdown(attr_breakdown)
        
        # 高等级数量奖励：每个5级以上属性额外+200分
        high_level_bonus = high_level_count * 200
        
        return float(base_power + level_bonus + high_level_bonus)

    def _get_appropriate_score(self, attr_breakdown: Dict[str, int], target_requirements: Optional[List[Tuple[str, int]]] = None, prioritize_high_level: bool = True) -> float:
        """
        根据上下文选择合适的评分方式
        
        Args:
            attr_breakdown: 属性分布
            target_requirements: 目标需求列表，如果提供则使用需求评分
            prioritize_high_level: 是否优先高等级属性
        
        Returns:
            适当的评分
        """
        if target_requirements:
            # 有目标需求时使用需求评分
            return self.requirement_score_from_attr_breakdown(attr_breakdown, target_requirements)
        elif prioritize_high_level:
            # 优先高等级属性时使用高等级评分
            return self.high_level_score_from_attr_breakdown(attr_breakdown)
        else:
            # 默认使用战力评分
            return float(self.combat_power_from_attr_breakdown(attr_breakdown))
        
    

    def requirement_score_from_attr_breakdown(self, attr_breakdown: Dict[str, int], target_requirements: List[Tuple[str, int]]) -> float:
        """
        改进的需求最佳方案评分函数
        
        评分逻辑：
        1. 目标属性覆盖度权重（最高优先级）
        2. 精确匹配奖励（第二优先级） - 刚好达到目标等级得最高分
        3. 接近程度评分（第三优先级） - 越接近目标等级分数越高
        4. 超出惩罚机制（第四优先级） - 超出目标等级开始扣分
        5. 基础战力分数（最低优先级）
        
        Args:
            attr_breakdown: 属性分布 {attr_name: total_points}
            target_requirements: 目标需求列表 [(attr_name, target_level), ...]
        
        Returns:
            需求评分，越高越好
        """
        if not target_requirements:
            # 如果没有目标要求，就按战力分排序
            return float(self.combat_power_from_attr_breakdown(attr_breakdown))
        
        score = 0.0
        
        # 1. 目标属性覆盖度权重（最高优先级）：100万分/个
        coverage_bonus = 0
        covered_targets = 0
        for attr_name, target_level in target_requirements:
            if attr_name in attr_breakdown:
                coverage_bonus += 1000000
                covered_targets += 1
        
        # 2. 精确匹配和接近程度评分（第二优先级）：每个目标最高10万分
        precision_bonus = 0
        total_distance_penalty = 0
        
        for attr_name, target_level in target_requirements:
            if attr_name in attr_breakdown:
                actual_points = attr_breakdown[attr_name]
                
                # 计算实际等级
                actual_level = 0
                for i, t in enumerate(self.thresholds):
                    if actual_points >= t:
                        actual_level = i + 1
                    else:
                        break
                
                # 计算等级差距
                level_diff = actual_level - target_level
                
                if level_diff == 0:
                    # 精确匹配：满分10万
                    precision_bonus += 100000
                elif level_diff > 0:
                    # 超出目标等级：开始扣分
                    # 超出1级扣20%，超出2级扣50%，超出3级扣80%，超出4级以上扣95%
                    if level_diff == 1:
                        precision_bonus += 80000  # 扣20%
                    elif level_diff == 2:
                        precision_bonus += 50000  # 扣50%
                    elif level_diff == 3:
                        precision_bonus += 20000  # 扣80%
                    else:
                        precision_bonus += 5000   # 扣95%
                        
                    # 额外的超出惩罚
                    total_distance_penalty += (level_diff - 1) * 10000
                        
                else:  # level_diff < 0，未达到目标等级
                    # 未达到目标：按接近程度给分
                    # 如果目标是6级，实际4级，给分 = 4/6 * 100000 = 66667
                    if target_level > 0:
                        closeness_ratio = max(0, actual_level) / target_level
                        precision_bonus += int(closeness_ratio * 100000)
                        
                        # 未达到目标的距离惩罚
                        total_distance_penalty += abs(level_diff) * 5000
            else:
                # 完全没有该属性：0分，但有重大惩罚
                total_distance_penalty += 50000
        
        # 3. 完美匹配奖励（当所有目标都精确匹配时）
        perfect_match_bonus = 0
        if covered_targets == len(target_requirements):
            all_perfect = True
            for attr_name, target_level in target_requirements:
                if attr_name in attr_breakdown:
                    actual_points = attr_breakdown[attr_name]
                    actual_level = 0
                    for i, t in enumerate(self.thresholds):
                        if actual_points >= t:
                            actual_level = i + 1
                        else:
                            break
                    
                    if actual_level != target_level:
                        all_perfect = False
                        break
            
            if all_perfect:
                perfect_match_bonus = 500000  # 50万分的完美匹配奖励
        
        # 4. 基础战力分数（最低优先级）
        base_power = self.combat_power_from_attr_breakdown(attr_breakdown)
        
        # 最终评分 = 覆盖度 + 精确匹配分 + 完美匹配奖励 - 距离惩罚 + 基础战力
        final_score = coverage_bonus + precision_bonus + perfect_match_bonus - total_distance_penalty + base_power
        
        return float(final_score)

    # 属性桶（UI）
    def prefilter_ui_mods(self, ui_mods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        先去掉"含排斥属性的模块"，再按属性建桶（不为被排斥属性建桶），
        每桶按该模块在该属性上的点数降序取前 bucket_cap；最后合并去重。
        """
        # 先整体过滤掉含排斥属性的模块
        base = [g for g in ui_mods if not self._ui_mod_has_excluded_attr(g)]
        if not base:
            return []

        # 按属性建桶
        buckets: Dict[str, List[Tuple[Dict[str, Any], int]]] = {}
        for g in base:
            for name, val in self._ui_iter_attr_pairs(g):
                if name in self.exclude_set:
                    continue  # 不为排斥属性建桶
                buckets.setdefault(name, []).append((g, int(val)))

        candidate: List[Dict[str, Any]] = []
        seen = set()
        for name, arr in buckets.items():
            # 按该属性点数排序，取前 N
            arr.sort(key=lambda t: t[1], reverse=True)
            for g, _ in arr[: self.bucket_cap]:
                uid = self._get_mod_uuid(g)  # 修复：使用正确的方法名
                if uid not in seen:
                    seen.add(uid)
                    candidate.append(g)

        return candidate

    
    # 快速计算（多解版本）  
    def fast_multi_solutions(self, modules: List[Any], slots: Optional[int] = None, prioritize_high_level: bool = True, num_solutions: int = 30, max_attempts: int = 200) -> List[ModuleSolution]:
        """
        快速计算多个解：贪心+局部搜索生成多个不同的优质解
        
        Args:
            modules: 输入模块列表
            slots: 模块槽位数量
            prioritize_high_level: 是否优先高等级属性
            num_solutions: 目标解的数量
            max_attempts: 最大尝试次数
            
        Returns:
            按评分降序排列的解列表
        """
        if not modules:
            return []
            
        # 预过滤
        filtered_modules = self.prefilter_modules(modules)
        if len(filtered_modules) < (slots or self.slots):
            return []
        
        solutions: List[ModuleSolution] = []
        seen_combinations = set()
        
        attempts = 0
        while len(solutions) < num_solutions and attempts < max_attempts:
            attempts += 1
            
            # 贪心构造初始解
            initial_solution = self.greedy_construct_solution(
                filtered_modules, slots, prioritize_high_level
            )
            
            if initial_solution is None:
                continue
                
            # 局部搜索改进
            improved_solution = self.local_search_improve(
                initial_solution, filtered_modules, prioritize_high_level=prioritize_high_level
            )
            
            if improved_solution is None:
                continue
                
            # 去重检查
            combo_sig = tuple(sorted(self._get_mod_uuid(m) for m in improved_solution.modules))
            if combo_sig not in seen_combinations:
                seen_combinations.add(combo_sig)
                solutions.append(improved_solution)
        
        # 按分数降序排序
        solutions.sort(key=lambda x: x.score, reverse=True)
        return solutions

    def fast_multi_solutions_ui(self, ui_mods: List[Dict[str, Any]], slots: Optional[int] = None, prioritize_high_level: bool = True, num_solutions: int = 30, max_attempts: int = 200) -> List[ModuleSolution]:
        """
        快速计算多个解（UI版本）：贪心+局部搜索生成多个不同的优质解
        
        Args:
            ui_mods: UI模块字典列表
            slots: 模块槽位数量
            prioritize_high_level: 是否优先高等级属性
            num_solutions: 目标解的数量
            max_attempts: 最大尝试次数
            
        Returns:
            按评分降序排列的解列表
        """
        if not ui_mods:
            return []
            
        # 预过滤
        filtered_modules = self.prefilter_ui_mods(ui_mods)
        if len(filtered_modules) < (slots or self.slots):
            return []
        
        solutions: List[ModuleSolution] = []
        seen_combinations = set()
        
        attempts = 0
        while len(solutions) < num_solutions and attempts < max_attempts:
            attempts += 1
            
            # 贪心构造初始解
            initial_solution = self.greedy_construct_solution_ui(
                filtered_modules, slots, prioritize_high_level
            )
            
            if initial_solution is None:
                continue
                
            # 局部搜索改进
            improved_solution = self.local_search_improve_ui(
                initial_solution, filtered_modules, prioritize_high_level=prioritize_high_level
            )
            
            if improved_solution is None:
                continue
                
            # 去重检查
            combo_sig = tuple(sorted(self._get_mod_uuid(m) for m in improved_solution.modules))
            if combo_sig not in seen_combinations:
                seen_combinations.add(combo_sig)
                solutions.append(improved_solution)
        
        # 按分数降序排序
        solutions.sort(key=lambda x: x.score, reverse=True)
        return solutions

    def greedy_construct_solution_ui(self, ui_mods: List[Dict[str, Any]], slots: Optional[int] = None, prioritize_high_level: bool = True) -> Optional[ModuleSolution]:
        """基于 UI 模块字典的贪心初始解；可选择优先高等级属性。"""
        k = int(slots or self.slots)
        if len(ui_mods) < k:
            return None

        current: List[Dict[str, Any]] = [random.choice(ui_mods)]
        while len(current) < k:
            best_g = None
            best_score = None
            for g in ui_mods:
                if g in current:
                    continue
                test = current + [g]
                pts = self._ui_combo_to_pts(test)
                # 根据参数选择评分方式
                if prioritize_high_level:
                    s = self.high_level_score_from_attr_breakdown(pts)
                else:
                    s = self.combat_power_from_attr_breakdown(pts)
                if best_score is None or s > best_score:
                    best_g, best_score = g, s
            if best_g is None:
                break
            current.append(best_g)

        pts_final = self._ui_combo_to_pts(current)
        if prioritize_high_level:
            final_score = self.high_level_score_from_attr_breakdown(pts_final)
        else:
            final_score = self.combat_power_from_attr_breakdown(pts_final)
        return ModuleSolution(current, final_score, pts_final)

    def local_search_improve_ui(self, solution: Optional[ModuleSolution], all_ui_mods: List[Dict[str, Any]], max_iter: int = 30, sample_k: int = 24, prioritize_high_level: bool = True) -> Optional[ModuleSolution]:
        """基于 UI 模块字典的局部搜索：单点替换，可选择优先高等级属性。"""
        if solution is None:
            return None

        best = solution
        seen_solutions = set()
        def _combo_sig(mods):
            return tuple(sorted(self._get_mod_uuid(x) for x in mods))
        seen_solutions.add(_combo_sig(best.modules))

        for _ in range(max_iter):
            improved = False
            for i in range(len(best.modules)):
                pool = all_ui_mods
                if len(pool) > sample_k:
                    pool = random.sample(pool, sample_k)
                for g in pool:
                    # 避免把当前解里已有的模块（哪怕是不同对象但 uuid 一样）
                    if any(self._get_mod_uuid(g) == self._get_mod_uuid(x) for x in best.modules):
                        continue
                    new_mods = list(best.modules)
                    new_mods[i] = g
                    sig = _combo_sig(new_mods)
                    if sig in seen_solutions:
                        continue
                    pts = self._ui_combo_to_pts(new_mods)
                    # 根据参数选择评分方式
                    if prioritize_high_level:
                        s = self.high_level_score_from_attr_breakdown(pts)
                    else:
                        s = self.combat_power_from_attr_breakdown(pts)
                    if s > best.score:
                        best = ModuleSolution(new_mods, s, pts)
                        seen_solutions.add(sig)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return best

    
    # 枚举（UI）
    def enum_all_combinations_ui_memory_safe_dual_topk(
        self,
        ui_mods: List[Dict[str, Any]],
        slots: Optional[int] = None,
        prioritize_high_level: bool = True,
        remove_uuid_dedup: bool = False,
        progress_callback=None,
        max_keep: int = 200,
        target_requirements: Optional[List[Tuple[str, int]]] = None
    ) -> Tuple[List[ModuleSolution], List[ModuleSolution]]:
        """
        双Top-K枚举：同时维护战力最高和需求最佳两套排序结果
        
        Returns:
            Tuple[战力最高方案列表, 需求最佳方案列表]
        """
        k = int(slots or self.slots)
        valid = [g for g in ui_mods if not self._ui_mod_has_excluded_attr(g)]
        if len(valid) < k:
            return [], []

        from itertools import combinations
        
        total_combinations = comb(len(valid), k)
        
        if progress_callback:
            progress_callback(0, total_combinations, "开始双Top-K内存安全枚举...")
        
        # 两个独立的Top-K堆
        battle_power_heap = []  # (battle_power_score, tie, solution)
        requirement_heap = []   # (requirement_score, tie, solution)
        
        processed = 0
        last_update_time = time.time()
        last_gc_time = time.time()
        
        # 用于进度显示的最低分数
        min_battle_power = 0
        min_requirement_score = 0
        
        for combo in combinations(valid, k):
            combo_list = list(combo)
            
            pts = self._ui_combo_to_pts(combo_list)
            
            # 计算两种不同的评分
            battle_power = float(self.combat_power_from_attr_breakdown(pts))
            
            if target_requirements:
                requirement_score = self.requirement_score_from_attr_breakdown(pts, target_requirements)
            else:
                # 没有目标需求时，需求分数等于战力分数
                requirement_score = battle_power
            
            # 创建解决方案对象（注意：这里score字段根据用途设置不同值）
            battle_solution = ModuleSolution(combo_list, battle_power, pts)
            requirement_solution = ModuleSolution(combo_list, requirement_score, pts)
            
            # 维护战力Top-K堆
            if len(battle_power_heap) < max_keep:
                heapq.heappush(battle_power_heap, (battle_power, next(self._tie), battle_solution))
                if len(battle_power_heap) == 1:
                    min_battle_power = battle_power
                else:
                    min_battle_power = min(score for score, _, _ in battle_power_heap)
            elif battle_power > battle_power_heap[0][0]:
                heapq.heapreplace(battle_power_heap, (battle_power, next(self._tie), battle_solution))
                min_battle_power = min(score for score, _, _ in battle_power_heap)
            
            # 维护需求Top-K堆
            if len(requirement_heap) < max_keep:
                heapq.heappush(requirement_heap, (requirement_score, next(self._tie), requirement_solution))
                if len(requirement_heap) == 1:
                    min_requirement_score = requirement_score
                else:
                    min_requirement_score = min(score for score, _, _ in requirement_heap)
            elif requirement_score > requirement_heap[0][0]:
                heapq.heapreplace(requirement_heap, (requirement_score, next(self._tie), requirement_solution))
                min_requirement_score = min(score for score, _, _ in requirement_heap)
            
            processed += 1
            current_time = time.time()
            
            # 1秒更新一次进度
            if progress_callback and (current_time - last_update_time >= 1.0 or processed == total_combinations):
                percentage = int(100 * processed / total_combinations)
                progress_callback(processed, total_combinations, 
                            f"已处理 {processed:,}/{total_combinations:,} ({percentage}%) "
                            f"战力:{min_battle_power:.0f} 需求:{min_requirement_score:.0f}")
                last_update_time = current_time
            
            # 定期垃圾回收
            if current_time - last_gc_time > 3.0:
                gc.collect()
                last_gc_time = current_time
        
        # 提取并排序两套结果
        battle_results = [sol for score, _tie, sol in battle_power_heap]
        battle_results.sort(key=lambda s: s.score, reverse=True)
        
        requirement_results = [sol for score, _tie, sol in requirement_heap]
        requirement_results.sort(key=lambda s: s.score, reverse=True)
        
        return battle_results, requirement_results

    def enum_all_combinations_ui_threaded_memory_safe_dual_topk(
        self,
        ui_mods: List[Dict[str, Any]],
        slots: Optional[int] = None,
        prioritize_high_level: bool = True,
        remove_uuid_dedup: bool = False,
        progress_callback=None,
        cancel_flag=None,
        num_threads=None,
        max_keep: int = 200,
        target_requirements: Optional[List[Tuple[str, int]]] = None
    ) -> Tuple[List[ModuleSolution], List[ModuleSolution]]:
        """
        多线程双Top-K枚举版本
        
        Returns:
            Tuple[战力最高方案列表, 需求最佳方案列表]
        """
        import threading
        import queue
        from concurrent.futures import ThreadPoolExecutor
        from itertools import combinations
        import multiprocessing
        
        k = int(slots or self.slots)
        valid = [g for g in ui_mods if not self._ui_mod_has_excluded_attr(g)]
        if len(valid) < k:
            return [], []

        total_combinations = comb(len(valid), k)
        
        if num_threads is None:
            num_threads = min(4, multiprocessing.cpu_count())
        
        if progress_callback:
            progress_callback(0, total_combinations, f"启动{num_threads}个线程进行双Top-K枚举...")
        
        # 全局双Top-K堆（线程安全）
        global_lock = threading.Lock()
        global_battle_heap = []
        global_requirement_heap = []
        processed_count = [0]
        count_lock = threading.Lock()
        
        def process_combinations_batch_dual(combo_batch, thread_id):
            """每个线程处理一批组合，维护两个局部Top-K"""
            local_battle_heap = []
            local_requirement_heap = []
            local_max_keep = max_keep // num_threads + 50
            
            for combo in combo_batch:
                if cancel_flag and cancel_flag.is_set():
                    return
                
                combo_list = list(combo)
                pts = self._ui_combo_to_pts(combo_list)
                
                # 计算两种评分
                battle_power = float(self.combat_power_from_attr_breakdown(pts))
                
                if target_requirements:
                    requirement_score = self.requirement_score_from_attr_breakdown(pts, target_requirements)
                else:
                    requirement_score = battle_power
                
                battle_solution = ModuleSolution(combo_list, battle_power, pts)
                requirement_solution = ModuleSolution(combo_list, requirement_score, pts)
                
                # 维护线程局部双Top-K
                if len(local_battle_heap) < local_max_keep:
                    heapq.heappush(local_battle_heap, (battle_power, next(self._tie), battle_solution))
                elif battle_power > local_battle_heap[0][0]:
                    heapq.heapreplace(local_battle_heap, (battle_power, next(self._tie), battle_solution))
                
                if len(local_requirement_heap) < local_max_keep:
                    heapq.heappush(local_requirement_heap, (requirement_score, next(self._tie), requirement_solution))
                elif requirement_score > local_requirement_heap[0][0]:
                    heapq.heapreplace(local_requirement_heap, (requirement_score, next(self._tie), requirement_solution))
                
                # 更新全局计数
                with count_lock:
                    processed_count[0] += 1
            
            # 合并到全局堆
            with global_lock:
                # 合并战力堆
                for score, _tie, sol in local_battle_heap:
                    if len(global_battle_heap) < max_keep:
                        heapq.heappush(global_battle_heap, (score, next(self._tie), sol))
                    elif score > global_battle_heap[0][0]:
                        heapq.heapreplace(global_battle_heap, (score, next(self._tie), sol))
                
                # 合并需求堆
                for score, _tie, sol in local_requirement_heap:
                    if len(global_requirement_heap) < max_keep:
                        heapq.heappush(global_requirement_heap, (score, next(self._tie), sol))
                    elif score > global_requirement_heap[0][0]:
                        heapq.heapreplace(global_requirement_heap, (score, next(self._tie), sol))
        
        # 分批处理
        all_combinations = combinations(valid, k)
        batch_size = max(100, total_combinations // (num_threads * 10))
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            current_batch = []
            batch_count = 0
            
            for combo in all_combinations:
                if cancel_flag and cancel_flag.is_set():
                    break
                    
                current_batch.append(combo)
                
                if len(current_batch) >= batch_size:
                    future = executor.submit(process_combinations_batch_dual, current_batch, batch_count)
                    futures.append(future)
                    current_batch = []
                    batch_count += 1
            
            if current_batch:
                future = executor.submit(process_combinations_batch_dual, current_batch, batch_count)
                futures.append(future)
            
            # 等待完成并更新进度
            while futures:
                completed_futures = [f for f in futures if f.done()]
                for future in completed_futures:
                    futures.remove(future)
                    try:
                        future.result()
                    except Exception as e:
                        print(f"线程执行错误: {e}")
                
                # 更新进度显示
                if progress_callback:
                    current_processed = processed_count[0]
                    with global_lock:
                        min_battle = min(score for score, _, _ in global_battle_heap) if global_battle_heap else 0
                        min_req = min(score for score, _, _ in global_requirement_heap) if global_requirement_heap else 0
                            
                    progress_callback(current_processed, total_combinations, 
                                f"多线程处理 {current_processed:,}/{total_combinations:,} "
                                f"战力:{min_battle:.0f} 需求:{min_req:.0f}")
                
                if futures:
                    time.sleep(1.0)
        
        # 提取最终结果
        with global_lock:
            battle_results = [sol for score, _tie, sol in global_battle_heap]
            requirement_results = [sol for score, _tie, sol in global_requirement_heap]
        
        battle_results.sort(key=lambda s: s.score, reverse=True)
        requirement_results.sort(key=lambda s: s.score, reverse=True)
        
        return battle_results, requirement_results

    def calculate_solution_score(self, modules: List[Any], prioritize_high_level: bool = False) -> Tuple[float, Dict[str, int]]:
        """
        通用评分入口：
          - 如果传入的是 UI 字典列表：直接聚合并评分
          - 如果是拥有 .parts（且 part 有 .name/.value）的对象列表：按对象聚合后评分
        """
        if not modules:
            return 0.0, {}
        if isinstance(modules[0], dict):
            pts = self._ui_combo_to_pts(modules)  # type: ignore[arg-type]
            if prioritize_high_level:
                score = self.high_level_score_from_attr_breakdown(pts)
            else:
                score = float(self.combat_power_from_attr_breakdown(pts))
            return score, pts
        pts: Dict[str, int] = {}
        for m in modules:
            parts = getattr(m, "parts", [])
            for p in parts:
                name = getattr(p, "name", None)
                val = int(getattr(p, "value", 0))
                if name is None:
                    continue
                pts[name] = pts.get(name, 0) + val
        if prioritize_high_level:
            score = self.high_level_score_from_attr_breakdown(pts)
        else:
            score = float(self.combat_power_from_attr_breakdown(pts))
        return score, pts

    def calculate_combat_power(self, modules: List[Any]) -> Tuple[int, Dict[str, int]]:
        """通用战斗力入口（同上自动识别）。"""
        if not modules:
            return 0, {}
        if isinstance(modules[0], dict):
            pts = self._ui_combo_to_pts(modules)  # type: ignore[arg-type]
            return self.combat_power_from_attr_breakdown(pts), pts
        pts: Dict[str, int] = {}
        for m in modules:
            parts = getattr(m, "parts", [])
            for p in parts:
                name = getattr(p, "name", None)
                val = int(getattr(p, "value", 0))
                if name is None:
                    continue
                pts[name] = pts.get(name, 0) + val
        return self.combat_power_from_attr_breakdown(pts), pts

    def prefilter_modules(self, modules: List[Any]) -> List[Any]:
        """
        通用属性桶入口：
          - UI 字典：走 prefilter_ui_mods
          - ModuleInfo：尽力按 .parts 聚合到类似 UI 的流程（仍然会先过滤掉含排斥属性）
        """
        if not modules:
            return []
        if isinstance(modules[0], dict):
            return self.prefilter_ui_mods(modules)  # type: ignore[arg-type]

        # 兼容 ModuleInfo：把含排斥属性的对象过滤，然后按属性建桶
        attr_buckets: Dict[str, List[Tuple[Any, int]]] = {}
        base = []
        # 先过滤掉含排斥属性的
        for m in modules:
            parts = getattr(m, "parts", [])
            if any(getattr(p, "name", None) in self.exclude_set for p in parts):
                continue
            base.append(m)

        for m in base:
            parts = getattr(m, "parts", [])
            for p in parts:
                name = getattr(p, "name", None)
                val = int(getattr(p, "value", 0))
                if name is None or name in self.exclude_set:
                    continue
                attr_buckets.setdefault(name, []).append((m, val))

        candidate: List[Any] = []
        seen = set()
        for name, arr in attr_buckets.items():
            arr.sort(key=lambda t: t[1], reverse=True)
            for obj, _ in arr[: self.bucket_cap]:
                oid = id(obj)
                if oid not in seen:
                    seen.add(oid)
                    candidate.append(obj)
        return candidate

    def greedy_construct_solution(self, modules: List[Any], slots: Optional[int] = None, prioritize_high_level: bool = True) -> Optional[ModuleSolution]:
        """通用贪心（自动识别 UI/ModuleInfo），可选择优先高等级属性，最终返回的 modules 为你传入的同构对象列表。"""
        k = int(slots or self.slots)
        if not modules or len(modules) < k:
            return None

        # 为了通用，内部评分一律转 pts 再算
        current = [random.choice(modules)]
        while len(current) < k:
            best = None
            best_score = None
            for g in modules:
                if g in current:
                    continue
                test = current + [g]
                score, pts = self.calculate_solution_score(test, prioritize_high_level)
                if best_score is None or score > best_score:
                    best, best_score = g, score
            if best is None:
                break
            current.append(best)

        final_score, pts_final = self.calculate_solution_score(current, prioritize_high_level)
        return ModuleSolution(current, final_score, pts_final)

    def local_search_improve(self, solution: Optional[ModuleSolution], all_modules: List[Any],max_iter: int = 30, sample_k: int = 24, prioritize_high_level: bool = True) -> Optional[ModuleSolution]:
        """通用局部搜索：单点替换，可选择优先高等级属性。"""
        if solution is None:
            return None

        best = solution
        # 记录已见解：使用按 uuid 排序后的组合签名
        seen_solutions = set()
        def _sig(mods: List[Any]):
            return tuple(sorted(self._get_mod_uuid(x) for x in mods))
        seen_solutions.add(_sig(best.modules))

        for _ in range(max_iter):
            improved = False
            for i in range(len(best.modules)):
                pool = all_modules
                if len(pool) > sample_k:
                    pool = random.sample(pool, sample_k)
                for g in pool:
                    # 避免把"同一模块（不同实例）"换入
                    if any(self._get_mod_uuid(g) == self._get_mod_uuid(x) for x in best.modules):
                        continue
                    new_mods = list(best.modules)
                    new_mods[i] = g
                    sig = _sig(new_mods)
                    if sig in seen_solutions:
                        continue

                    s, pts = self.calculate_solution_score(new_mods, prioritize_high_level)
                    if s > best.score:
                        best = ModuleSolution(new_mods, s, pts)
                        seen_solutions.add(sig)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return best

    def enum_all_combinations(
        self,
        modules: List[Any],
        slots: Optional[int] = None,
        prioritize_high_level: bool = True,
        remove_uuid_dedup: bool = False,
    ) -> List[ModuleSolution]:
        """
        通用枚举（可接受 UI 模块字典或含 .parts 的对象）；
        自动过滤含排斥属性的模块；返回前 **按战斗力分数降序排序**。
        """
        k = int(slots or self.slots)
        if not modules:
            return []

        # 过滤含排斥属性
        def has_excluded(m: Any) -> bool:
            if isinstance(m, dict):
                for name, _ in self._ui_iter_attr_pairs(m):
                    if name in self.exclude_set:
                        return True
                return False
            parts = getattr(m, "parts", None)
            if parts:
                for p in parts:
                    name = getattr(p, "name", None)
                    if name in self.exclude_set:
                        return True
            return False

        valid = [m for m in modules if not has_excluded(m)]
        if len(valid) < k:
            return []

        sols: List[ModuleSolution] = []
        # 目前去重逻辑保持与原版一致（如需开启，把下面注释解开即可）
        seen = set() if not remove_uuid_dedup else None

        from itertools import combinations
        for combo in combinations(valid, k):
            combo_list = list(combo)

            # 可选：基于模块 UUID 的组合去重（保持默认关闭）
            # if seen is not None:
            #     sig = tuple(sorted(self._get_mod_uuid(x) for x in combo_list))
            #     if sig in seen:
            #         continue
            #     seen.add(sig)

            score, pts = self.calculate_solution_score(combo_list, prioritize_high_level)
            sols.append(ModuleSolution(combo_list, score, pts))

        # ★ 只按战斗力分数降序排序（score 越大越靠前）
        sols.sort(key=lambda s: float(s.score), reverse=True)
        return sols
