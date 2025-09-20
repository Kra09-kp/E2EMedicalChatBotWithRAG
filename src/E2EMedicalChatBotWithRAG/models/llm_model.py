from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.E2EMedicalChatBotWithRAG.logger import logger
from src.E2EMedicalChatBotWithRAG.utils import load_env_variable,get_prompt_text
from src.E2EMedicalChatBotWithRAG.config.configuration import ConfigurationManager
from src.E2EMedicalChatBotWithRAG.exceptions import AppException
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler


# print("LLMModel.py is loaded")
load_env_variable("GROQ_API_KEY")

class LLMAssistant:
    def __init__(self,config=ConfigurationManager()):
        """
        Initialize the LLMAssistant with configuration parameters.
        """
        try:
            self.config = config.get_chatbot_config()
        except Exception as e:
            logger.error(f"Error in ConfigurationManager: {e}")
            raise AppException(e) from e
    
    def get_model(self):
        """
        Returns the configured language model.

        This function initializes and returns a ChatGroq model with the specified parameters.

        Returns:
            ChatGroq: The configured language model.
        """
        try:
            handler = AsyncIteratorCallbackHandler()
            llm = ChatGroq(
                model=self.config.llm_model_name,
                temperature=0.7,
                max_tokens=512,
                streaming=True,                # 🔑 enable streaming
                                        )
            return llm
        except Exception as e:
            logger.error(f"Error initializing llm model: {e}")
            raise AppException(e) from e
    
    def get_template(self):
        """
        Returns the chat template used for generating prompts.

        This function defines a chat template that includes a user message placeholder.

        Returns:
            ChatPromptTemplate: The chat prompt template.
        """
        system_prompt = get_prompt_text(prompt_path=self.config.system_prompt_path)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", "{input}")
            ]
        )
        return prompt
