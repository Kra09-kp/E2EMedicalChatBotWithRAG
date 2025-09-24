from src.E2EMedicalChatBotWithRAG.logger import logger
from src.E2EMedicalChatBotWithRAG.vectorestores import RedisDB
from src.E2EMedicalChatBotWithRAG.preprocess import DocumentPreprocesser
from src.E2EMedicalChatBotWithRAG.exceptions import AppException
from langchain.schema import Document


def main():
    try:
        redis_client = RedisDB()
        document_preprocesser = DocumentPreprocesser()
        preprocessed_data = document_preprocesser.run()
        redis_client.create_vector_store(preprocessed_data)
        new_doc = Document(
            page_content="This project is built by Kirti Pogra, she used langchain and groq and RAG functionalitize.\
                And this project is for her portfolio. \
                The code is available on GitHub. The project name is medical chatbot using rag.\
                Kirti Pogra is persuing MCA in Artificial Intelligence and Machine Learning.\
                She is trying to build her career in the field of Deep Learning and GenAI.",
            metadata={"source": "kirti pogra"}
        )
        redis_client.add_document_to_store(new_doc)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise AppException(e)


import time

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"This took around {(time.time()-start_time)/60} minutes")