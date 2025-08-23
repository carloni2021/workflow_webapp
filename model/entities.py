from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class JobRecord:
    id: int
    class_id: str = "Class1"
    arrival_time: float = 0.0
    completion_time: Optional[float] = None
    visit_times: Dict[str, float] = field(default_factory=dict)  # tempo alla stazione (PS)
    wait_times: Dict[str, float] = field(default_factory=dict)   # 0 in PS (placeholder per futuri FCFS)
