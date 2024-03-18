from fastapi import APIRouter
from pydantic import BaseModel

class chatbotQuery(BaseModel):
    user_query: str

router = APIRouter()

@router.post("/chatbot/query")
async def chatbotQuery(body: chatbotQuery):
    return True
    # return await 
