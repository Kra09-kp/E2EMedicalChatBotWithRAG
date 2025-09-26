from E2EMedicalChatBotWithRAG.logger import logger
from E2EMedicalChatBotWithRAG.utils import load_env_variable
from E2EMedicalChatBotWithRAG.config.configuration import ConfigurationManager
from E2EMedicalChatBotWithRAG.exceptions import AppException
from E2EMedicalChatBotWithRAG.models.embedding_model import EmbeddingModel
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

pinecone_api_key = load_env_variable("PINECONE_API_KEY")

class PineconeDB(EmbeddingModel):
    """ 
    A class for interacting with Pinecone vector database.
    """
    def __init__(self,config=ConfigurationManager()):
        try:
            self.config = config.get_chatbot_config()
            self.index_name = self.config.index_name
            self.dimension = self.config.dimension  # Dimension of the embedding model
            self._init_connection()
            super().__init__()
            self.embedding_model = self._get_model()

        except Exception as e:
            logger.error(f"Error in ConfigurationManager: {e}")
            raise AppException(e) from e
        
    def get_retriever(self):
        """
        Load a retriever for similarity search on an existing Pinecone index.

        This method only loads; it will not create or populate the index.
        If the index is missing, a warning is logged and an AppException is raised.

        Returns:
            langchain.vectorstores.base.VectorStoreRetriever:
                Retriever configured for cosine similarity with k=3.

        Raises:
            AppException: If the index does not exist or cannot be loaded.
        """
        try:
            if not self.pinecone_client.has_index(self.index_name):
                logger.warning(
                    f"Pinecone index '{self.index_name}' not found. "
                    "You must call create_vector_store_and_retriever()."
                )
            vector_store =  PineconeVectorStore.from_existing_index(
                embedding=self.embedding_model,
                index_name=self.index_name,
            )
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        except Exception as e:
            raise AppException(e)
        else:
            return retriever


    def create_vector_store_and_retriever(self,chunked_text,embedding_model):
        """
        Creates a new Pinecone vector store from the given list of documents.

        This function takes in a list of documents, an embedding model, and an index name.
        It then creates a new Pinecone vector store from the documents, using the specified embedding model and index name.

        If there is an error during the creation of the vector store, an AppException is raised.
        Otherwise, the created vector store is returned.

        Args:
            chunked_text (List[Document]): A list of documents to create the vector store from.
            embedding_model (Any): The embedding model to use for creating the vector store.

        Returns:
            PineconeVectorStore: The created vector store.
        """
        try:
            self._create_index()

            # create a new Pinecone vector store from the documents
            doc_vector_store = PineconeVectorStore.from_documents(
                documents=chunked_text,  # list of documents to create the vector store from
                embedding=embedding_model,  # the embedding model to use for creating the vector store
                index_name=self.index_name,  # the name of the index to use for the vector store
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
        Adds a new document to the Pinecone vector store.

        Args:
            doc_vector_store (PineconeVectorStore): The Pinecone vector store to add the document to.
            new_doc (Document): The document to add to the vector store.

        Raises:
            AppException: If there is an error during the addition of the document to the vector store.
        """
        try:
            doc_vector_store.add_documents(documents=[new_doc])
        except Exception as e:
            raise AppException(e)
        
        logger.info(f"Added new document to Pinecone vector store")

    def get_index(self):
        index = self.pinecone_client.Index(self.index_name)
        return index



    def _init_connection(self):
        try:
            self.pinecone_client = Pinecone(api_key=pinecone_api_key)
        except Exception as e:
            raise AppException(e)
    
    def _create_index(self):
        if not self.pinecone_client.has_index(self.index_name):
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=self.dimension,  # Dimension of the embedding model
                metric="cosine", 
                spec=ServerlessSpec(
                    cloud="aws",  # or "gcp" depending on your setup
                    region="us-east-1"  # choose the region you want
                )
            )
            logger.info(f"Created Pinecone index: {self.index_name}")

