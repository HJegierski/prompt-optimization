from dataclasses import dataclass
from enum import Enum


class Preference(str, Enum):
    LHS = "LHS"
    RHS = "RHS"
    NEITHER = "Neither"


@dataclass(frozen=True)
class Product:
    id: str
    name: str
    description: str
    class_name: str
    category_hierarchy: str
    grade: int
