from dataclasses import dataclass
from pathlib import Path

@dataclass
class ChatBotConfig:
    data_path: Path
    embedding_model_name: str
    system_prompt_path: Path
    llm_model_name: str