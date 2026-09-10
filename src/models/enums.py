from enum import Enum


class DefectType(str, Enum):
    SCRATCH = "scratch"
    DENT = "dent"
    CRACK = "crack"
    DISCOLORATION = "discoloration"
    BUBBLE = "bubble"
    STAIN = "stain"
    CHIP = "chip"
    NONE = "none"


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InspectionStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
