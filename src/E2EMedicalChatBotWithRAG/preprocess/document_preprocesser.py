from src.E2EMedicalChatBotWithRAG.logger import logger
from src.E2EMedicalChatBotWithRAG.exceptions import AppException
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from src.E2EMedicalChatBotWithRAG.config.configuration import ConfigurationManager


class DocumentPreprocesser:
    def __init__(self,config=ConfigurationManager()):
        try:
            self.config = config.get_chatbot_config()
        except Exception as e:
            logger.error(f"Error in ConfigurationManager: {e}")
            raise AppException(e) from e
        
    def run(self,doc_path=None):
        """
        Main entry point for the DocumentPreprocesser class.

        This function loads documents from the configured data path,
        filters out any documents that don't meet the specified criteria,
        and then chunks the documents into smaller chunks for processing.

        Returns:
            List[Document]: The list of preprocessed documents.
        """
        try:
            if doc_path is None:
                doc_path = self.config.data_path
            # Load documents from the configured data path
            documents = self._load_documents(str(doc_path))
            logger.info(f"Loaded {len(documents)} documents from {doc_path}")

            # Filter out any documents that don't meet the specified criteria
            documents = self._filter_documents(documents)
            logger.info(f"Filtered out {len(documents)} documents that don't meet the criteria")

            # Chunk the documents into smaller chunks for processing
            documents = self._chunk_documents(documents)
            logger.info(f"Chunked {len(documents)} documents into smaller chunks.")
        except Exception as e:
            raise AppException(e)
        else:
            return documents

    def _load_documents(self, doc_path: str) -> List[Document]:
        try:
            loader = DirectoryLoader(
                doc_path,
                glob="*.pdf",
                loader_cls=PyPDFLoader #type:ignore
            )
            documents = loader.load()
        except Exception as e:
            raise AppException(e)
        else:
            return documents

    def _filter_documents(self, docs: List[Document]) -> List[Document]:
        minimal_docs: List[Document] = []
        try:
            for doc in docs:
                src = doc.metadata.get("source", "unknown")
                minimal_doc = Document(
                    page_content=doc.page_content,
                    metadata={"source": src}
                )
                minimal_docs.append(minimal_doc)
        except Exception as e:
            raise AppException(e)
        else:
            return minimal_docs


    def _chunk_documents(self, docs:List[Document], chunk_size:int=1000, chunk_overlap:int=200) -> List[Document]:
        try:
            # Initialize the text splitter with the specified chunk size and overlap
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        
            # Split the documents into chunks using the text splitter
            chunked_docs = text_splitter.split_documents(docs)

        except Exception as e:
            raise AppException(e)
        else:
            return chunked_docs
