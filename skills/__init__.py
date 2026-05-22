"""
JARVIS Skill System

技能系統：在 prompt 送給模型前，先比對使用者輸入是否匹配已註冊的技能。
匹配成功則執行技能並將結果注入 prompt 上下文。

用法：
    from skills import SkillRegistry
    registry = SkillRegistry()
    registry.register(date_skill)
    prompt, matched = registry.process("今天天氣如何")
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger("jarvis.skills")


class Skill:
    """單一技能定義。"""
    name: str
    keywords: List[str]
    handler: Callable[[str], Optional[str]]

    def __init__(self, name: str, keywords: List[str], handler: Callable):
        self.name = name
        self.keywords = keywords
        self.handler = handler


class SkillRegistry:
    """技能註冊中心。"""

    def __init__(self):
        self._skills: List[Skill] = []

    def register(self, skill: Skill):
        self._skills.append(skill)
        logger.info(f"Skill 已註冊: {skill.name}")

    def process(self, user_input: str) -> Tuple[str, bool]:
        """
        比對並執行技能。

        Args:
            user_input: 使用者原始輸入

        Returns:
            (injected_prompt, matched) — 若有匹配技能則 injected_prompt 含技能結果
        """
        for skill in self._skills:
            matched = any(kw in user_input for kw in skill.keywords)
            if not matched:
                continue
            try:
                result = skill.handler(user_input)
                if result:
                    logger.info(f"Skill 命中: {skill.name}")
                    return result, True
            except Exception as e:
                logger.warning(f"Skill {skill.name} 執行失敗: {e}")
        return user_input, False
