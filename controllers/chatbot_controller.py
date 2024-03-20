from fastapi import APIRouter
from pydantic import BaseModel
from chatbot import *

class chatbotQuery(BaseModel):
    user_query: str

router = APIRouter()

@router.post("/chatbot/query")
def chatbotQuery(body: chatbotQuery):
    return crypto_chatbot(body.user_query)
