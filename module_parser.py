"""
模组解析器
"""

import json
from typing import Dict, List, Optional, Any
from BlueProtobuf_pb2 import CharSerialize
from module_types import (
    ModuleInfo, ModulePart, ModuleType, ModuleAttrType, ModuleCategory,
    MODULE_NAMES, MODULE_ATTR_NAMES, MODULE_CATEGORY_MAP
)


class ModuleParser:
    """模组解析器"""
    
    def __init__(self):
        pass
    
    def parse_module_info(self, v_data: CharSerialize, category: str = "全部", attributes: List[str] = None, 
                         exclude_attributes: List[str] = None, match_count: int = 1):
        """
        解析模组信息（用于UI集成）

        Args:
            v_data: VData数据
            category: 模组类型（攻击/守护/辅助/全部）
            attributes: 要筛选的属性词条列表
            exclude_attributes: 要排除的属性词条列表
            match_count: 模组需要包含的指定词条数量
            
        Returns:
            解析后的模组列表
        """
        
        mod_infos = v_data.Mod.ModInfos

        modules = []
        for package_type, package in v_data.ItemPackage.Packages.items():
            for key, item in package.Items.items():
                if item.HasField('ModNewAttr') and item.ModNewAttr.ModParts:
                    config_id = item.ConfigId
                    module_name = MODULE_NAMES.get(config_id, f"未知模组({config_id})")
                    mod_parts = list(item.ModNewAttr.ModParts)
                    # 查找模组详细信息
                    mod_info = mod_infos.get(key) if mod_infos else None

                    if mod_info and hasattr(mod_info, 'InitLinkNums'):
                        module_info = ModuleInfo(
                            name=module_name,
                            config_id=config_id,
                            uuid=item.Uuid,
                            quality=item.Quality,
                            parts=[]
                        )

                        init_link_nums = mod_info.InitLinkNums
                        for i, part_id in enumerate(mod_parts):
                            if i < len(init_link_nums):
                                attr_name = MODULE_ATTR_NAMES.get(part_id, f"未知属性({part_id})")
                                attr_value = init_link_nums[i]
                                module_part = ModulePart(
                                    id=part_id,
                                    name=attr_name,
                                    value=attr_value
                                )
                                module_info.parts.append(module_part)
                        modules.append(module_info)
                else:
                    # 不是模组背包
                    break
        
        if modules:
            print(f"解析到 {len(modules)} 个模组")
            
            # 属性筛选（如果有筛选条件）
            if attributes or exclude_attributes:
                filtered_modules = self._filter_modules_by_attributes(
                    modules, attributes, exclude_attributes, match_count
                )
                print(f"筛选后剩余 {len(filtered_modules)} 个模组")
                return filtered_modules
        
        return modules
    
    def _filter_modules_by_attributes(self, modules: List[ModuleInfo], attributes: List[str] = None, 
                                     exclude_attributes: List[str] = None, match_count: int = 1) -> List[ModuleInfo]:
        """根据属性词条筛选模组
        
        Args:
            modules: 模组列表
            attributes: 要筛选的属性词条列表
            exclude_attributes: 要排除的属性词条列表
            match_count: 模组需要包含的指定词条数量
            
        Returns:
            筛选后的模组列表
        """
        filtered_modules = []
        
        for module in modules:
            # 获取模组的所有属性名称
            module_attrs = [part.name for part in module.parts]
            
            # 检查是否包含排除的属性
            if exclude_attributes:
                has_excluded_attr = any(attr in exclude_attributes for attr in module_attrs)
                if has_excluded_attr:
                    continue
            
            # 检查包含的属性数量
            if attributes:
                matching_attrs = [attr for attr in module_attrs if attr in attributes]
                if len(matching_attrs) < match_count:
                    continue
            
            filtered_modules.append(module)
        
        return filtered_modules