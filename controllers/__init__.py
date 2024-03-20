from fastapi import APIRouter
from controllers.chatbot_controller import router as chatbotRouter

apis = APIRouter()

apis.include_router(chatbotRouter)
__all__ = ["apis"]