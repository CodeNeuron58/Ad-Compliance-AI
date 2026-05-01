from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: str
    pinecone_api_key: str
    
    # LangSmith tracing
    langchain_api_key: str
    langchain_tracing_v2: bool
    langchain_project: str
    
settings = Settings()
