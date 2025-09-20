from src.E2EMedicalChatBotWithRAG.logger import logger
from src.E2EMedicalChatBotWithRAG.chains.rag_chain import RAGChain
from src.E2EMedicalChatBotWithRAG.exceptions import AppException

rag_chain = RAGChain()

async def main():
    try:
       
        question = input("Enter your medical question: ")
        print(f"Question: {question}")

        print(f"Answer: \n")
        async for token in rag_chain.ainvoke(question):
            print(token, end="", flush=True)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise AppException(e) from e

import asyncio

if __name__ == "__main__":
    print("Hello, Welcome to E2EMedicalChatBotWithRAG!")
    asyncio.run(main())