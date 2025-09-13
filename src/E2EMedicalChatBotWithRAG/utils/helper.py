import yaml
import sys,os
from src.E2EMedicalChatBotWithRAG.exceptions import AppException
from src.E2EMedicalChatBotWithRAG.logger import logger
from pathlib import Path

def read_yaml_file(file_path: str) -> dict:
    """
    Read a YAML file and return its contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, 'rb') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise AppException(e) from e 

def get_prompt_text(prompt_path: Path) -> str:
    """
    Read a prompt text file and return its contents as a string.
    prompt_path: str
    """
    try:
        with open(prompt_path, 'r') as file:
            return file.read()
    except Exception as e:
        raise AppException(e) from e

def load_env_variable(var_name:str,set_env:bool=True) -> str:
    """Load environment variable from .env file and optionally set it in os.environ
    
    Args:
        var_name (str): Name of the environment variable to load
        set_env (bool, optional): Whether to set the variable in os.environ. Defaults to True.
    
    Returns:
        str: Value of the environment variable
    """
    from dotenv import load_dotenv
    load_dotenv()  # take environment variables from .env.
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environment variable {var_name} not found")
    if set_env:
        os.environ[var_name] = value
    logger.info(f"Loaded environment variable: {var_name}")
    return value

