from fastapi import APIRouter
from pydantic import BaseModel
from ..chatbot import crypto_chatbot

class chatbotQuery(BaseModel):
    user_query: str

router = APIRouter()

@router.post("/chatbot/query")
async def chatbotQuery(body: chatbotQuery):
    return await crypto_chatbot(body.user_query)
    # return await 
