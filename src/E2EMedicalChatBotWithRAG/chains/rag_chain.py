from src.E2EMedicalChatBotWithRAG.logger import logger
from src.E2EMedicalChatBotWithRAG.models.llm_model import LLMAssistant
from src.E2EMedicalChatBotWithRAG.vectorestores import RedisDB, PineconeDB, AsyncPineconeDB
from src.E2EMedicalChatBotWithRAG.exceptions import AppException
from langchain.schema.runnable import RunnablePassthrough

class RAGChain:
    def __init__(self,use_redis=False,sync=False):
        """
        Synchronous initialization for RedisDB or PineconeDB.
        """
        self.llm_assistant = LLMAssistant()
        if sync:
            if use_redis:
                self.vector_store = RedisDB()
            else:
                self.vector_store = PineconeDB()
        
            self.chain = self._create_chain()
        self.achain = None
        self.avector_store = None

    
    @classmethod
    async def make_async(cls, client):
        """
        Async constructor for AsyncPineconeDB.

        Returns:
            RAGChain instance with async chain initialized.
        """
        self = cls()
        self.achain = await self._create_async_chain(client)
        return self

    def invoke(self, question: str):
        """
        Synchronous call to RAG chain.
        """
        
        try:
            for token in self.chain.stream(question):
                yield token.content
        except Exception as e:
            raise AppException(e) from e

    async def ainvoke(self, question: str):
        """
        Asynchronous call to RAG chain.
        """
        
        try:
            async for token in self.achain.astream(question): # type: ignore
                yield token.content
        except Exception as e:
            raise AppException(e) from e

    def _create_chain(self):
        """
        Build synchronous RAG chain.
        """
        try:
            llm = self.llm_assistant.get_model()
            prompt = self.llm_assistant.get_template()
            retriever = self.vector_store.get_retriever()
            rag_chain = (
                {
                    "context": retriever,
                    "input": RunnablePassthrough()
                }
                | prompt
                | llm
            )
            return rag_chain
        except Exception as e:
            logger.error(f"Error in creating sync RAG chain: {e}")
            raise AppException(e) from e

    async def _create_async_chain(self,client):
        """
        Build asynchronous RAG chain.
        """
        try:
            llm = self.llm_assistant.get_model()
            prompt = self.llm_assistant.get_template()
            retriever = await AsyncPineconeDB(client).get_retriever()
            rag_chain = (
                {
                    "context": retriever,
                    "input": RunnablePassthrough()
                }
                | prompt
                | llm
            )
            return rag_chain
        except Exception as e:
            logger.error(f"Error in creating async RAG chain: {e}")
            raise AppException(e) from e
