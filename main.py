from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from controllers import apis

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        print("exc......",{exc})
        # pushLogs(exc)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=jsonable_encoder({"success":False, "statusCode": 400, "message": exc._errors[0]["loc"][1] + " is required"}),
        )
    except Exception as e:
        print("err......",{e})
        # pushLogs(e)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=jsonable_encoder({"success":False, "statusCode": 400, "message": "Payload is Empty"}),
        )

@app.get("/")
def root():
    return {"message": "server is running +0k..." }


app.include_router(apis, prefix="/api")

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)