"""03_xxx.py처럼 숫자로 시작하는 파일은 import로 못 불러와서, 파일 경로로 직접 불러오는 도우미."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def load_module(relpath: str) -> ModuleType:
    """예: load_module('modeling/09BoardingProbability.py'). 한 번 불러온 파일은 다시 안 읽음."""
    path = ROOT / relpath
    name = "m_" + path.stem
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
