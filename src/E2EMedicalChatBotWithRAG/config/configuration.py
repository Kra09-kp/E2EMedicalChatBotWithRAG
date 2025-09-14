import os
import sys
from src.E2EMedicalChatBotWithRAG.entity.config_entity import ChatBotConfig
from src.E2EMedicalChatBotWithRAG.exceptions import AppException
from src.E2EMedicalChatBotWithRAG.logger import logger
from src.E2EMedicalChatBotWithRAG.constants import *
from src.E2EMedicalChatBotWithRAG.utils import read_yaml_file

class ConfigurationManager:
    def __init__(self,config_file_path:str = CONFIG_FILE_PATH):
        try:
            self.config = read_yaml_file(file_path=config_file_path)
        except Exception as e:
            raise AppException(e) from e
    
    def get_chatbot_config(self) -> ChatBotConfig:
        try:
            chatbot_config = self.config['chatbot_config']
            config = ChatBotConfig(
                data_path=chatbot_config['DATA_PATH'],
                embedding_model_name=chatbot_config['EMBEDDING_MODEL_NAME'],
                system_prompt_path=chatbot_config['SYSTEM_PROMPT_PATH'],
                llm_model_name=chatbot_config['LLM_MODEL_NAME'],
                index_name=chatbot_config['INDEX_NAME'],
                dimension=chatbot_config['DIMENSION']
            )
            return config
        except Exception as e:
            logger.error(f"Error in getting chatbot config: {e}")
            raise AppException(e) from e