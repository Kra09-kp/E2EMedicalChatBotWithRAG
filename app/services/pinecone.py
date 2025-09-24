from pinecone import PineconeAsyncio
from src.E2EMedicalChatBotWithRAG.utils import load_env_variable

class PineCone:
    def __init__(self):
        self.api_key = load_env_variable("PINECONE_API_KEY")

    def init(self):
        self.pc = PineconeAsyncio(api_key=self.api_key)
        return self.pc
    
    async def close(self):
        await self.pc.close()