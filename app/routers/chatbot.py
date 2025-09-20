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

            # Step 3: stream the answer token-by-token
#             sample_response = """**Garlic-Butter Veggie & Egg Fried Rice**
# (*Feeds 1–2 people*)

# **Ingredients**

# * 1 cup cooked rice (leftover rice from the fridge is perfect)
# * 1 cup mixed veggies (frozen peas, carrots, capsicum, corn—whatever’s handy)
# * 2 eggs
# * 2 tbsp butter (or ghee/oil)
# * 3–4 garlic cloves, finely chopped
# * 1–2 tbsp soy sauce (or a splash of lemon + salt if you don’t have it)
# * Black pepper & salt to taste
# * Optional: chopped spring onion or coriander for garnish

# Would you like a vegetarian alternative without eggs, or a different cuisine (like quick pasta or Indian tadka khichdi)?
# """
            
#             # inside your websocket handler
#             for word in sample_response:
#                 await websocket.send_text(word)
#                 await asyncio.sleep(0.005)  # tiny pause for “typing” effect (optional)

            async for token in rag_chain.ainvoke(question):
                await websocket.send_text(token)

            # Step 4: optionally signal completion
            await websocket.send_text("[[END]]")

    except Exception as e:
        print("WebSocket error:", e)
    finally:
        await websocket.close()


