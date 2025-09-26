from E2EMedicalChatBotWithRAG.logger import logger
from E2EMedicalChatBotWithRAG.utils import load_env_variable
from E2EMedicalChatBotWithRAG.config.configuration import ConfigurationManager
from E2EMedicalChatBotWithRAG.exceptions import AppException
from E2EMedicalChatBotWithRAG.models import EmbeddingModel
from E2EMedicalChatBotWithRAG.retrievers import PineconeAsyncRetriever
from pinecone import ServerlessSpec, PineconeAsyncio
from langchain_pinecone import PineconeVectorStore


class AsyncPineconeDB(EmbeddingModel):
    """ 
    A class for interacting with Pinecone vector database.
    """
    def __init__(self,client,config=ConfigurationManager()):
        try:
            self.config = config.get_chatbot_config()
            self.index_name = self.config.index_name
            self.dimension = self.config.dimension  # Dimension of the embedding model
            super().__init__()
            self.pinecone_client = client
            

        except Exception as e:
            logger.error(f"Error in ConfigurationManager: {e}")
            raise AppException(e) from e
        
    async def get_retriever(self,k=3):
        """
        Load a retriever for similarity search on an existing Pinecone index.

        This method only loads; it will not create or populate the index.
        If the index is missing, a warning is logged and an AppException is raised.

        Returns:
            PineconeAsyncRetriever: The configured retriever for similarity search.
        Raises:
            AppException: If the index does not exist or cannot be loaded.
        """
        try:
            if not await self.pinecone_client.has_index(self.index_name):
                logger.warning(
                    f"Pinecone index '{self.index_name}' not found. "
                    "You must call create_vector_store_and_retriever()."
                )
            
            retriever = PineconeAsyncRetriever(embedding_model=EmbeddingModel(),
                                            index=await self.get_index(),
                                            k=k)
        except Exception as e:
            raise AppException(e)
        else:
            return retriever


    async def create_vector_store_and_retriever(self,chunked_text):
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
            await self._create_index()
            await self.set_embedding_model()
            logger.info(f"Creating Pinecone vector store with index: {self.index_name}")
            logger.info(f"Using embedding model: {self.embedding_model}")

            # create a new Pinecone vector store from the documents
            await PineconeVectorStore.afrom_documents(
                documents=chunked_text,  # list of documents to create the vector store from
                embedding=self.embedding_model,  # the embedding model to use for creating the vector store
                index_name=self.index_name,  # the name of the index to use for the vector store
            )

            retriever = PineconeAsyncRetriever(embedding_model=self.embedding_model,
                                            index= await self.get_index(),
                                            k=3)

        except Exception as e:
            # if there is an error during the creation of the vector store, raise an AppException
            raise AppException(e)
        else:
            # otherwise, return the created vector store
            return retriever

    async def add_document_to_store(self,doc_vector_store,new_doc):
        """
        Adds a new document to the Pinecone vector store.

        Args:
            doc_vector_store (PineconeVectorStore): The Pinecone vector store to add the document to.
            new_doc (Document): The document to add to the vector store.

        Raises:
            AppException: If there is an error during the addition of the document to the vector store.
        """
        try:
            await doc_vector_store.aadd_documents(documents=[new_doc])
        except Exception as e:
            raise AppException(e)
        
        logger.info(f"Added new document to Pinecone vector store")

    async def get_index(self):
        """
        You can find host name by calling get_list_of_indexes() function
        """
        host_name = await self.get_host_name()
        index = self.pinecone_client.IndexAsyncio(host=host_name)
        return index

    async def get_host_name(self):
        index_info = await self.pinecone_client.describe_index(self.index_name)
        host_name = index_info.host
        return host_name


    async def _init_connection(self):
        try:
            pinecone_api_key = load_env_variable("PINECONE_API_KEY")
            self.pinecone_client = PineconeAsyncio(api_key=pinecone_api_key)
        except Exception as e:
            raise AppException(e)
        
    
    async def _create_index(self):
        if not await self.pinecone_client.has_index(self.index_name):
            await self.pinecone_client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,  # Dimension of the embedding model
                    metric="cosine", 
                    spec=ServerlessSpec(
                        cloud="aws",  # or "gcp" depending on your setup
                        region="us-east-1"  # choose the region you want
                    )
                )
            logger.info(f"Created Pinecone index: {self.index_name}")

    async def get_list_of_indexes(self):
            # Do async things
            index_list = await self.pinecone_client.list_indexes()
            return index_list

    async def close(self):
        await self.pinecone_client.close()

    async def set_embedding_model(self):
        if not self.embedding_model:
            self.embedding_model = self._get_model()
        return self
