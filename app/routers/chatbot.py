from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from E2EMedicalChatBotWithRAG.chains.rag_chain import RAGChain
from E2EMedicalChatBotWithRAG.logger import logger
from contextlib import asynccontextmanager
from app.services import PineCone


class Question(BaseModel):
    question: str

@asynccontextmanager
async def lifespan(router: APIRouter):
    logger.info("Starting the Medical Chatbot")
    try:
        pc = PineCone()
        
        client = pc.init()
        global rag_chain
        rag_chain = await RAGChain.make_async(client=client)
        logger.info("pinecone client initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing pinecone client: {e}")
        raise e

    yield

    try:
        await client.close()
        logger.info("pinecone client closed successfully")
    except Exception as e:
        logger.error(f"Error closing pinecone client: {e}")
    logger.info("Medical Chatbot is shutting down")

router = APIRouter(lifespan=lifespan,
                   tags=["chatbot"]
                   )

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
            

            # Step 3: stream the response tokens back to the client
            async for token in rag_chain.ainvoke(question):
                # print(token)
                await websocket.send_text(token)

            # Step 4: optionally signal completion
            await websocket.send_text("[[END]]")

    except WebSocketDisconnect:
        # client closed the connection – just exit
        pass

    except Exception as e:
        print("WebSocket error:", e)
        
    finally:
        # close only if it’s still open
        if websocket.client_state.name != "DISCONNECTED":
            await websocket.close()


