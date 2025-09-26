from dataclasses import dataclass
from pathlib import Path

@dataclass
class ChatBotConfig:
    data_path: Path
    embedding_model_name: str
    embedding_model_url: str    
    system_prompt_path: Path
    llm_model_name: str
    index_name: str
    redis_url: str
    dimension: int
