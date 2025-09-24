from fastapi import FastAPI,Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.routers import chatbot
from fastapi.responses import HTMLResponse

app = FastAPI()


templates = Jinja2Templates(directory="./app/templates")

app.include_router(chatbot.router)
app.mount("/static", StaticFiles(directory="./app/templates/static", html=True), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/about")
async def about():
    return {"message": "About the E2E Medical ChatBot"}