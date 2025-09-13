from src.E2EMedicalChatBotWithRAG.logger import logger
from src.E2EMedicalChatBotWithRAG.utils import load_env_variable
from src.E2EMedicalChatBotWithRAG.config.configuration import ConfigurationManager
from src.E2EMedicalChatBotWithRAG.exceptions import AppException
from langchain_huggingface import HuggingFaceEmbeddings
import torch


class EmbeddingModel:
    def __init__(self,config=ConfigurationManager()):
        try:
            self.config = config.get_chatbot_config()
        except Exception as e:
            logger.error(f"Error in ConfigurationManager: {e}")
            raise AppException(e) from e
    
    def embed(self,doc):
        
        pass


    def _get_model(self):
        try:
            model_name = self.config.embedding_model_name
            model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
            logger.info(f"Successfully loaded embedding model: {model_name}")
            return embedding
        except Exception as e:
            logger.error(f"Error in loading embedding model: {e}")
            raise AppException(e) from e
    
    