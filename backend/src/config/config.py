from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "LLM Embedding Explorer"
    admin_email: str = "admin@example.com"
    default_model_name: str = "all-MiniLM-L6-v2"
    default_model_description: str = "Default Sentence Transformer Model (all-MiniLM-L6-v2)"

    class Config:
        env_file = ".env"

settings = Settings() 