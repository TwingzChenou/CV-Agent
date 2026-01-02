from fastapi import FastAPI
from app.core.config import settings

app = FastAPI(
    title="Project-CV Agent API",
    version="1.0.0",
    description="Backend API for Project-CV Agent"
)

@app.get("/")
async def root():
    return {
        "message": "Welcome to Project-CV Agent API",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
