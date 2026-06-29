"""
=============================================================
CrystalEvolutionAgent

core/candidate.py

Author:
    Wu Jinming
    ChatGPT

Description
-----------
Core data structure used throughout the whole project.

A CrystalCandidate represents ONE crystal during the entire
evolution process.

Unlike a pymatgen Structure, it stores

    • crystal structure
    • evolution history
    • evaluation result
    • mutation history
    • planner decisions
    • metadata

Every module (Mutation / Evaluator / Reflection / Planner /
Population) should operate on CrystalCandidate rather than
Structure directly.

=============================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4
from copy import deepcopy

from pymatgen.core import Structure


@dataclass
class CrystalCandidate:
    """
    Evolution object of one crystal.

    Parameters
    ----------
    structure
        pymatgen Structure

    source
        Origin of this candidate.

        Examples
        --------
        "materials_project"

        "mutation"

        "crossover"

        "llm"

    generation
        Current generation index.
    """

    # ==========================================================
    # Basic Information
    # ==========================================================

    structure: Structure

    source: str = "unknown"

    generation: int = 0

    # ==========================================================
    # Automatically generated ID
    # ==========================================================

    candidate_id: str = field(default_factory=lambda: str(uuid4()))

    parent_id: str | None = None

    # ==========================================================
    # Evolution
    # ==========================================================

    mutation_history: list[dict] = field(default_factory=list)

    planner_history: list[dict] = field(default_factory=list)

    reflection_history: list[str] = field(default_factory=list)

    # ==========================================================
    # Validation
    # ==========================================================

    valid: bool = False

    validator_result: dict[str, Any] = field(default_factory=dict)

    # ==========================================================
    # Evaluation
    # ==========================================================

    evaluated: bool = False

    evaluator_result: dict[str, Any] = field(default_factory=dict)

    fitness: float | None = None

    # ==========================================================
    # Metadata
    # ==========================================================

    metadata: dict[str, Any] = field(default_factory=dict)

    # ==========================================================
    # Initialization
    # ==========================================================

    def __post_init__(self):

        self.formula = self.structure.composition.reduced_formula

        self.num_atoms = len(self.structure)

        self.composition = self.structure.composition

    # ==========================================================
    # Utilities
    # ==========================================================

    def clone(self) -> "CrystalCandidate":
        """
        Deep copy candidate.

        Used by mutation/crossover.
        """

        new_candidate = deepcopy(self)

        new_candidate.candidate_id = str(uuid4())

        new_candidate.parent_id = self.candidate_id

        new_candidate.generation += 1

        return new_candidate

    # ----------------------------------------------------------

    def add_mutation(
        self,
        mutation_name: str,
        parameters: dict
    ) -> None:
        """
        Record mutation history.
        """

        self.mutation_history.append({

            "mutation": mutation_name,

            "parameters": parameters

        })

    # ----------------------------------------------------------

    def add_planner_record(
        self,
        planner_output: dict
    ) -> None:

        self.planner_history.append(planner_output)

    # ----------------------------------------------------------

    def add_reflection(
        self,
        reflection: str
    ) -> None:

        self.reflection_history.append(reflection)

    # ----------------------------------------------------------

    def set_validation(
        self,
        valid: bool,
        result: dict
    ) -> None:

        self.valid = valid

        self.validator_result = result

    # ----------------------------------------------------------

    def set_evaluation(
        self,
        result: dict,
        fitness: float
    ) -> None:

        self.evaluated = True

        self.evaluator_result = result

        self.fitness = fitness

    # ----------------------------------------------------------

    def summary(self) -> dict:
        """
        Lightweight information for logging.
        """

        return {

            "candidate_id": self.candidate_id,

            "formula": self.formula,

            "generation": self.generation,

            "fitness": self.fitness,

            "valid": self.valid,

            "evaluated": self.evaluated,

            "num_atoms": self.num_atoms,

            "source": self.source

        }

    # ----------------------------------------------------------

    def __repr__(self):

        return (

            f"CrystalCandidate("
            f"{self.formula}, "
            f"Gen={self.generation}, "
            f"Fitness={self.fitness})"

        )
