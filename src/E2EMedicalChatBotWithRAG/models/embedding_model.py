from src.E2EMedicalChatBotWithRAG.logger import logger
from src.E2EMedicalChatBotWithRAG.utils import load_env_variable
from src.E2EMedicalChatBotWithRAG.config.configuration import ConfigurationManager
from src.E2EMedicalChatBotWithRAG.exceptions import AppException
from langchain_huggingface import HuggingFaceEmbeddings
from src.E2EMedicalChatBotWithRAG.preprocess import DocumentPreprocesser
import torch


class EmbeddingModel:
    def __init__(self,config=ConfigurationManager()):
        try:
            self.config = config.get_chatbot_config()
        except Exception as e:
            logger.error(f"Error in ConfigurationManager: {e}")
            raise AppException(e) from e
    
    def embed(self,doc):
        try:
            preprocessor = DocumentPreprocesser()
            processed_doc = preprocessor.run(doc)
            texts = [doc.page_content for doc in processed_doc]
            embeddings = self._get_model().embed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error in embedding document: {e}")
            raise AppException(e) from e

    def _get_model(self):
        """
        Loads the configured HuggingFace embedding model.

        This function loads the configured HuggingFace embedding model and returns it.
        If there is an error during the loading of the embedding model, an AppException is raised.

        Returns:
            HuggingFaceEmbeddings: The loaded embedding model.
        """
        try:
            model_name = self.config.embedding_model_name
            model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
            logger.info(f"Successfully loaded embedding model: {model_name}")
            return embedding
        except Exception as e:
            logger.error(f"Error in loading embedding model: {e}")
            raise AppException(e) from e
    
    