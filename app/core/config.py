from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    GOOGLE_API_KEY: str = Field(..., description="API Key for Google services (Gemini, etc)")
    PINECONE_API_KEY: str = Field(..., description="API Key for Pinecone Vector DB")
    PINECONE_INDEX: str = Field(..., description="Name of the Pinecone Index")
    GITHUB_TOKEN: str = Field(..., description="GitHub Token for API access")
    PORT: int = Field(8000, description="Port to listen on (Render provides this)")

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
