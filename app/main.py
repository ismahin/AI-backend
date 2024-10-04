from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router

app = FastAPI(title="Crop Disease Classification and Chatbot API")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend's URL and port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router that handles both disease classification and chatbot
app.include_router(api_router, prefix="/api")

# Start the server (optional if deploying)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
