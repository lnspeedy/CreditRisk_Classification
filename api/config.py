from pydantic import BaseSettings

class Settings(BaseSettings):
    """
    Read environment variables (case-insensitive)
    Example: HOST variable will be read for host.
    """

    host: str
    port: int
    training_data_folder: str
    model_folder: str
    preprocess_pipe_folder: str

settings = Settings()

