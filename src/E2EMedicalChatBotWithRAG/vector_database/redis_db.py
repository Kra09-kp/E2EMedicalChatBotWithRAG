from src.E2EMedicalChatBotWithRAG.logger import logger
from src.E2EMedicalChatBotWithRAG.config.configuration import ConfigurationManager
from src.E2EMedicalChatBotWithRAG.exceptions import AppException
from src.E2EMedicalChatBotWithRAG.models.embedding_model import EmbeddingModel
from langchain_redis import RedisVectorStore


class RedisDB(EmbeddingModel):
    """ 
    A class for interacting with Redis vector database.
    """
    def __init__(self,config=ConfigurationManager()):
        try:
            super().__init__()

            self.config = config.get_chatbot_config()
            self.index_name = self.config.index_name
            self.dimension = self.config.dimension  # Dimension of the embedding model
            self.redis_url = self.config.redis_url
            self.embedding_model = self._get_model()

            self._init_connection()

        except Exception as e:
            logger.error(f"Error in ConfigurationManager: {e}")
            raise AppException(e) from e
        
    def get_retriever(self):
        """
        Load a retriever for similarity search on an existing Redis index.

        This method only loads; it will not create or populate the index.
        If the index is missing, a warning is logged and an AppException is raised.

        Returns:
            langchain.vectorstores.base.VectorStoreRetriever:
                Retriever configured for cosine similarity with k=3.

        Raises:
            AppException: If the index does not exist or cannot be loaded.
        """
        try:
            vector_store =  self.redis_client.from_existing_index(
                embedding=self.embedding_model,
                index_name=self.index_name,
                redis_url=self.redis_url
            )
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            sample_data = retriever.invoke("what is acne?")
            if len(sample_data) < 3:
                logger.warning(f"Index {self.index_name} is empty or does not exist.")
                
        except Exception as e:
            raise AppException(e)
        else:
            return retriever


    def create_vector_store_and_retriever(self,chunked_text,embedding_model):
        """
        Creates a new redis vector store from the given list of documents.

        This function takes in a list of documents, an embedding model, and an index name.
        It then creates a new redis vector store from the documents, using the specified embedding model and index name.

        If there is an error during the creation of the vector store, an AppException is raised.
        Otherwise, the created vector store is returned.

        Args:
            chunked_text (List[Document]): A list of documents to create the vector store from.
            embedding_model (Any): The embedding model to use for creating the vector store.

        Returns:
            redisVectorStore: The created vector store.
        """
        try:
            # create a new redis vector store from the documents
            doc_vector_store = self.redis_client.from_documents(
                documents=chunked_text,  # list of documents to create the vector store from
                embedding=embedding_model,  # the embedding model to use for creating the vector store
                index_name=self.index_name,  # the name of the index to use for the vector store
                redis_url=self.redis_url

            )
            retriever = doc_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        except Exception as e:
            # if there is an error during the creation of the vector store, raise an AppException
            raise AppException(e)
        else:
            # otherwise, return the created vector store
            return retriever

    def add_document_to_store(self,doc_vector_store,new_doc):
        """
        Adds a new document to the redis vector store.

        Args:
            doc_vector_store (RedisVectorStore): The redis vector store to add the document to.
            new_doc (Document): The document to add to the vector store.

        Raises:
            AppException: If there is an error during the addition of the document to the vector store.
        """
        try:
            doc_vector_store.add_documents(documents=[new_doc])
        except Exception as e:
            raise AppException(e)
        
        logger.info(f"Added new document to redis vector store")




    def _init_connection(self):
        try:
            self.redis_client = RedisVectorStore(
                index_name=self.index_name,
                redis_url = self.redis_url,
                embeddings=self.embedding_model
                )
        except Exception as e:
            raise AppException(e)
    
   