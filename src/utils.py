from langchain_core.prompts import ChatPromptTemplate
from typing import Any, Optional
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
import pandas as pd
import json
from loguru import logger
import os

def load_dataset_jsonl(
    dataset_name: str,
    dataset_path: str,
    max_samples: Optional[int] = 10,
) -> pd.DataFrame:
    """Load a JSONL dataset file."""
    logger.info(f"Loading dataset {dataset_name} from path {dataset_path}")

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset {dataset_name} not found at path {dataset_path}")
        return None

    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    if data:
        df = pd.DataFrame(data)
        if max_samples is not None:
            df = df.iloc[:max_samples]
        return df

    else:
        return None

def init_llm(
        system_prompt: str,
        model: str,
        model_provider: str,
        structured_output: Optional[BaseModel] = None,
        kwargs: Optional[dict] = None,
) -> Any:
    
    system_prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{query}")
    ])
    llm_target = init_chat_model(
        model=model,
        model_provider=model_provider,
        **kwargs
    )
    if structured_output is not None:
        llm_target = llm_target.with_structured_output(structured_output)
    llm_target = system_prompt_template | llm_target

    return llm_target


def init_embedding_model(
        model: str,
        model_provider: str,
        kwargs: dict,
) -> Any:
    embedding_model = init_embedding_model(
        model=model,
        model_provider=model_provider,
        **kwargs
    )
    return embedding_model