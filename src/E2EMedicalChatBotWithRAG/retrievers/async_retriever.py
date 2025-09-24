from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import PrivateAttr
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)


class PineconeAsyncRetriever(BaseRetriever):
    """
    A production-ready async retriever for Pinecone + HuggingFace embeddings.

    Parameters
    ----------
    embedding_model : Any
        Must expose an `embed_query(text: str) -> List[float]` method.
    index : Any
        Pinecone index client with an async `query(...)` method.
    k : int, default 3
        Number of documents to retrieve.
    search_kwargs : dict, optional
        Extra parameters for Pinecone's query call.
    tags : list[str], optional
        Custom tags for observability/monitoring.
    """

    _embedding_model: Any = PrivateAttr()
    _index: Any = PrivateAttr()

    k: int = 3
    search_kwargs: Dict[str, Any] = {}
    tags: Optional[List[str]] = None

    def __init__(self,
        embedding_model: Any,
        index: Any,
        k: int = 3,
        search_kwargs: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        
        super().__init__()
        self._embedding_model = embedding_model
        self._index = index
        self.k = k
        self.search_kwargs = search_kwargs or {"k": k}
        self.tags = tags or ["PineconeVectorStore", "HuggingFaceEmbeddings"]

    # Disable sync usage to force async pattern
    def _get_relevant_documents(self,query: str,*,run_manager: CallbackManagerForRetrieverRun,) -> List[Document]:

        raise NotImplementedError(
            "Synchronous retrieval is disabled. Use `aget_relevant_documents`."
        )

    async def _aget_relevant_documents(self,query: str,*,run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        
        """
        Retrieve top-k documents asynchronously from Pinecone.

        Returns
        -------
        list of langchain.schema.Document
        """
        # 1. Embed the query
        try:
            query_vector = self._embedding_model.embed_query(query)
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}") from e

        # 2. Query Pinecone
        try:
            response = await self._index.query(
                vector=query_vector,
                top_k=self.k,
                include_metadata=True,
                **self.search_kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Pinecone query failed: {e}") from e

        # 3. Build Document list
        documents: List[Document] = []
        for match in response.get("matches", []):
            metadata = match.get("metadata", {})
            text = metadata.get("text")
            if not text:
                continue  # Skip if no text found
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": metadata.get("source"),
                        "similarity_score": match.get("score"),
                    },
                )
            )

        return documents
