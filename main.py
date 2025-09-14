from src.E2EMedicalChatBotWithRAG.logger import logger
from src.E2EMedicalChatBotWithRAG.chains.rag_chain import RAGChain
from src.E2EMedicalChatBotWithRAG.exceptions import AppException

rag_chain = RAGChain()

def main():
    try:
       
        question = input("Enter your medical question: ")
        answer = rag_chain.invoke(question)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise AppException(e) from e

if __name__ == "__main__":
    print("Hello, Welcome to E2EMedicalChatBotWithRAG!")
    main()