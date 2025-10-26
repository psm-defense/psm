from typing import Optional
from pydantic import BaseModel, Field
import yaml
from src.defense.psm.psm import PSMConfig


class Config(BaseModel):
    llms: Optional[dict] = Field(
        description="Dictionary of LLM configurations for different models",
        default=None
    )

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            yaml_data = yaml.safe_load(f)
            return cls.model_validate(yaml_data)
