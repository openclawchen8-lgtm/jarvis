"""
Digital Human — 模式定義與組態 (T043)

支援兩種驅動模式：
  - BODY: 全身 3D VRM（Three.js + Mixamo 動畫, T042）
  - FACE: 臉部驅動（LivePortrait / LiteAvatar, T044 — 待完成）

可擴充：
  - HYBRID: 同時啟用臉部 + 身體（未來）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DigitalHumanMode(Enum):
    """數位人驅動模式。"""
    NONE = "none"       # 停用數位人
    BODY = "body"       # 全身 3D（Three.js VRM）
    FACE = "face"       # 臉部驅動（LivePortrait / LiteAvatar）
    HYBRID = "hybrid"   # 臉部 + 身體疊加（未來）


# 模式之間的依賴關係
MODE_DEPENDENCIES = {
    DigitalHumanMode.NONE: [],
    DigitalHumanMode.BODY: ["T042"],    # Three.js VRM
    DigitalHumanMode.FACE: ["T044"],    # MNN-LivePortrait (pending)
    DigitalHumanMode.HYBRID: ["T042", "T044"],
}

# 模式顯示名稱（前端用）
MODE_LABELS = {
    DigitalHumanMode.NONE: "無",
    DigitalHumanMode.BODY: "全身 3D",
    DigitalHumanMode.FACE: "臉部驅動",
    DigitalHumanMode.HYBRID: "全身 + 臉部",
}


@dataclass
class DHConfig:
    """數位人整合組態。"""
    mode: DigitalHumanMode = DigitalHumanMode.NONE
    body_model_path: str = "assets/models/jarvis_girl.glb"
    face_source_image: Optional[str] = None   # T044 用
    tts_enabled: bool = True
    viseme_enabled: bool = True
    a2bs_enabled: bool = True
    auto_command: bool = True                 # LLM 自動判斷動作指令

    def is_available(self) -> bool:
        """檢查模式是否可執行（依賴是否滿足）。"""
        deps = MODE_DEPENDENCIES.get(self.mode, [])
        if "T042" in deps:
            from pathlib import Path
            model = Path(self.body_model_path)
            if not model.exists():
                return False
        if "T044" in deps:
            # T044 not yet implemented
            return False
        return True

    def health(self) -> dict:
        return {
            "mode": self.mode.value,
            "label": MODE_LABELS.get(self.mode, "unknown"),
            "available": self.is_available(),
            "body_model": self.body_model_path,
            "face_source": self.face_source_image,
        }
