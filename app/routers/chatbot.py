from fastapi import APIRouter, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from E2EMedicalChatBotWithRAG.logger import logger
from src.E2EMedicalChatBotWithRAG.chains.rag_chain import RAGChain
from E2EMedicalChatBotWithRAG.exceptions import AppException
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import asyncio

class Question(BaseModel):
    question: str

router = APIRouter()
rag_chain = RAGChain()


@router.websocket("/ws/ask")
async def ask_ws(websocket: WebSocket):
    try:
        # Step 1: accept the connection
        await websocket.accept()
        while True:
            # Step 2: receive the client’s question (JSON or plain text)
            data = await websocket.receive_json()
            question = data.get("question")
            if not question:
                await websocket.send_text("[[ERROR]] No question provided.")
                continue
            if not question:
                await websocket.close(code=1003)
                return

            # Step 3: stream the response tokens back to the client
            for token in rag_chain.invoke(question):
                await websocket.send_text(token)

            # Step 4: optionally signal completion
            await websocket.send_text("[[END]]")

    except Exception as e:
        print("WebSocket error:", e)
    finally:
        await websocket.close()


