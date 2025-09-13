from src.E2EMedicalChatBotWithRAG.logger import logger
from src.E2EMedicalChatBotWithRAG.utils import load_env_variable
from src.E2EMedicalChatBotWithRAG.config.configuration import ConfigurationManager
from src.E2EMedicalChatBotWithRAG.exceptions import AppException
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

pinecone_api_key = load_env_variable("PINECONE_API_KEY")

class PineconeDB:
    """ 
    A class for interacting with Pinecone vector database.
    """
    def __init__(self,config=ConfigurationManager()):
        try:
            self.config = config.get_chatbot_config()
            self._init_connection()
            self.index = self._create_index()

        except Exception as e:
            logger.error(f"Error in ConfigurationManager: {e}")
            raise AppException(e) from e
    
    def create_vector_store_from_docs(self,chunked_text,embedding_model,index_name):
        """
        Creates a new Pinecone vector store from the given list of documents.

        This function takes in a list of documents, an embedding model, and an index name.
        It then creates a new Pinecone vector store from the documents, using the specified embedding model and index name.

        If there is an error during the creation of the vector store, an AppException is raised.
        Otherwise, the created vector store is returned.

        Args:
            chunked_text (List[Document]): A list of documents to create the vector store from.
            embedding_model (Any): The embedding model to use for creating the vector store.
            index_name (str): The name of the index to use for the vector store.

        Returns:
            PineconeVectorStore: The created vector store.
        """
        try:
            # create a new Pinecone vector store from the documents
            doc_vector_store = PineconeVectorStore.from_documents(
                documents=chunked_text,  # list of documents to create the vector store from
                embedding=embedding_model,  # the embedding model to use for creating the vector store
                index_name=index_name,  # the name of the index to use for the vector store
            )
        except Exception as e:
            # if there is an error during the creation of the vector store, raise an AppException
            raise AppException(e)
        else:
            # otherwise, return the created vector store
            return doc_vector_store
    
    def load_vector_store(self,embedding_model,index_name):
        """
        Loads an existing Pinecone vector store from the configured index.

        Returns:
            PineconeVectorStore: The loaded vector store.
        """
        try:
            # load existing index
            doc_vector_store = PineconeVectorStore.from_existing_index(
                embedding=embedding_model,
                index_name=index_name,
            )
        except Exception as e:
            raise AppException(e)
        else:
            return doc_vector_store
    
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



    def _init_connection(self):
        try:
            self.pinecone_client = Pinecone(api_key=pinecone_api_key)
        except Exception as e:
            raise AppException(e)
    
    def _create_index(self):
        index_name = "medical-chatbot"

        if not self.pinecone_client.has_index(index_name):
            self.pinecone_client.create_index(
                name=index_name,
                dimension=384,  # Dimension of the embedding model
                metric="cosine", 
                spec=ServerlessSpec(
                    cloud="aws",  # or "gcp" depending on your setup
                    region="us-east-1"  # choose the region you want
                )
            )
            logger.info(f"Created Pinecone index: {index_name}")
