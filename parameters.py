from datetime import datetime
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Tuple


class GameParameters(BaseModel):
    n_agents: int
    n_actions: int
    n_iter: int = 3000
    game_type: Literal["shapley", "sato", "conflict"]

    @model_validator(mode="after")
    def validate_game_actions(self):
        if self.game_type in {"sato", "shapley"} and self.n_actions != 3:
            raise ValueError(f"{self.game_type.capitalize()} requires 3 actions")

        return self


class NetworkParameters(BaseModel):
    network_type: Literal["er", "sbm"]
    p: float | None = None

    # SBM Parameters
    n_blocks: int | None = None
    q: int | None = None
    p_min: float | None = None
    p_max: float | None = None

    @model_validator(mode="after")
    def validate_sbm(self):
        if self.network_type == "sbm" and self.n_blocks is None:
            raise ValueError("n_blocks must be set for SBM experiments")

        if self.network_type == "sbm" and self.q is None:
            raise ValueError("q must be set for SBM experiments")

        return self


class ExperimentParameters(BaseModel):
    game_parameters: GameParameters
    network_parameters: NetworkParameters
    nP: int = (30,)
    nT: int = (30,)
    n_expt: int = (12,)
    p_range: Tuple[float, float] = (0.1, 1.0)
    T_range: Tuple[float, float] = (0.1, 3.5)
    n_refinements: int = 4
    timestamp: datetime = Field(default_factory=datetime.now)


class RunParameters(BaseModel):
    T: float
    n_agents: int
    n_actions: int
    n_iter: int
    game_type: Literal["sato", "shapley", "conflict"]
    n_expt: int
    network_parameters: NetworkParameters
