"""
模组解析器（仅解析；可选开启优化）
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any
from BlueProtobuf_pb2 import CharSerialize
from logging_config import get_logger
from module_types import (
    ModuleInfo, ModulePart, ModuleType, ModuleAttrType, ModuleCategory,
    MODULE_NAMES, MODULE_ATTR_NAMES, MODULE_CATEGORY_MAP
)

# 仅在“允许优化”时才用到优化器，避免无意义依赖
try:
    from module_optimizer import ModuleOptimizer  # 可选使用
except Exception:
    ModuleOptimizer = None  # 延迟检查

# 获取日志器
logger = get_logger(__name__)


class ModuleParser:
    """
    模组解析器：
    - 默认只负责把 VData 解析成 List[ModuleInfo]
    - 是否执行“组合优化/计算”由开关控制（默认 False）
    """

    def __init__(self, enable_optimize: bool = False):
        """
        Args:
            enable_optimize: 实例级开关。True 时允许在 parse 末尾触发内部优化。
        """
        self.logger = logger
        self.enable_optimize = bool(enable_optimize)

    def parse_module_info(
        self,
        v_data: CharSerialize,
        category: str = "攻击",
        attributes: Optional[List[str]] = None,
        do_optimize: Optional[bool] = None,
    ) -> List[ModuleInfo]:
        """
        解析模组信息（是否启用优化由开关决定；默认不优化）

        Args:
            v_data:        游戏抓包得到的 VData
            category:      模组类型（"攻击"/"守护"/"辅助"）
            attributes:    可选的属性白名单（仅保留所有词条都在名单内的模组）
            do_optimize:   本次调用是否执行优化。None 表示跟随实例开关；True/False 强制覆盖。

        Returns:
            List[ModuleInfo]: 仅解析后的模组对象列表
        """
        self.logger.info("开始解析模组（仅解析，默认不计算）")

        # ===== 解析 VData 中的模组信息 =====
        mod_infos = getattr(v_data, "Mod", None)
        mod_infos = getattr(mod_infos, "ModInfos", None)

        modules: List[ModuleInfo] = []

        # 遍历背包
        item_package = getattr(v_data, "ItemPackage", None)
        packages = getattr(item_package, "Packages", {}) if item_package else {}

        for package_type, package in packages.items():
            items = getattr(package, "Items", {})
            for key, item in items.items():
                # 仅处理包含 ModNewAttr 且有 ModParts 的条目
                if item.HasField("ModNewAttr") and getattr(item.ModNewAttr, "ModParts", None):
                    config_id = item.ConfigId
                    module_name = MODULE_NAMES.get(config_id, f"未知模组({config_id})")
                    mod_parts_ids = list(item.ModNewAttr.ModParts)

                    # 查找附加信息（如初始链接数）
                    mod_info = mod_infos.get(key) if mod_infos else None
                    init_link_nums = getattr(mod_info, "InitLinkNums", [])

                    module = ModuleInfo(
                        name=module_name,
                        config_id=config_id,
                        uuid=item.Uuid,
                        quality=item.Quality,
                        parts=[]
                    )

                    # 将部件 ID + 值 转为可读词条
                    for i, part_id in enumerate(mod_parts_ids):
                        if i < len(init_link_nums):
                            attr_name = MODULE_ATTR_NAMES.get(part_id, f"未知属性({part_id})")
                            attr_value = int(init_link_nums[i])
                            module.parts.append(ModulePart(id=part_id, name=attr_name, value=attr_value))

                    modules.append(module)
                else:
                    # 不是模组背包，或该条目没有模组属性，直接跳过
                    continue

        if not modules:
            self.logger.info("未解析到任何模组。")
            return []

        self.logger.debug(f"解析到 {len(modules)} 个模组信息")
        # 打印摘要（调试）
        for i, m in enumerate(modules[:10], 1):
            parts_str = ", ".join([f"{p.name}+{p.value}" for p in m.parts])
            self.logger.debug(f"  {i}. {m.name} ({parts_str})")
        if len(modules) > 10:
            self.logger.debug(f"... 其余 {len(modules) - 10} 个略")

        # ===== 可选：属性白名单过滤 =====
        if attributes:
            filtered_modules = self._filter_modules_by_attributes(modules, attributes)
            self.logger.info(f"属性筛选后剩余 {len(filtered_modules)} 个模组")
        else:
            filtered_modules = modules

        # ===== 可选：是否执行内部优化（默认 False） =====
        allow_opt = self.enable_optimize if do_optimize is None else bool(do_optimize)
        if allow_opt:
            self._optimize_module_combinations(filtered_modules, category)
        else:
            self.logger.info("已禁用优化：仅返回解析后的模组列表。")

        # 只返回解析结果，供 UI/上层逻辑使用
        return modules

    # -------- 内部辅助：属性筛选 --------
    def _filter_modules_by_attributes(self, modules: List[ModuleInfo], attributes: List[str]) -> List[ModuleInfo]:
        """
        根据属性词条列表进行“白名单”筛选：
        仅保留“所有词条都在 attributes 中”的模组
        """
        if not attributes:
            return modules

        attr_set = set(attributes)
        filtered: List[ModuleInfo] = []

        for m in modules:
            names = [p.name for p in m.parts]
            if all(n in attr_set for n in names):
                filtered.append(m)
                self.logger.debug(f"保留: '{m.name}' → {', '.join(names)}")
            else:
                invalid = [n for n in names if n not in attr_set]
                self.logger.debug(f"剔除: '{m.name}' 含未在白名单的词条: {', '.join(invalid)}")

        return filtered

    # -------- 内部辅助：可选优化 --------
    def _optimize_module_combinations(self, modules: List[ModuleInfo], category: str):
        """
        （可选）调用优化器做展示/打印。
        对自动导入到 UI 的流程不是必需；仅当你明确允许时才会执行。
        """
        try:
            if ModuleOptimizer is None:
                self.logger.warning("未找到 module_optimizer，跳过优化。")
                return

            category_map = {
                "攻击": ModuleCategory.ATTACK,
                "守护": ModuleCategory.GUARDIAN,
                "辅助": ModuleCategory.SUPPORT,
            }
            target_category = category_map.get(category, ModuleCategory.ATTACK)

            optimizer = ModuleOptimizer()
            optimizer.optimize_and_display(modules, target_category, top_n=20)
        except Exception as e:
            self.logger.error(f"模组搭配优化失败: {e}")
