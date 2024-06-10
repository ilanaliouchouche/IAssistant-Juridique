import os
import cohere

from typing import Dict, Generator, List, Tuple
import gradio as gr
from langchain_chroma import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()


class AIJurist:
    """
    AIJurist class to deploy a chatbot that answers legal questions based on
    the French civil code.
    """

    def __init__(self,
                 embedding_model_path: str,
                 vector_db_path: str,
                 sys_prompt_path: str,
                 top_k: int = 5,
                 embedding_model_name: str = os.getenv('EMBEDDING_MODEL_NAME'),
                 api_key: str = os.getenv('CO_API_KEY'),
                 device: str = 'cpu',
                 **kwargs):
        """
        Constructor for the AIJurist class.

        Parameters:
        embedding_model_path (str): Path to the embedding model.
        vector_db_path (str): Path to the vector database.
        sys_prompt_path (str): Path to the system prompt file.
        top_k (int): Number of documents to retrieve from the vector database.
        embedding_model_name (str): Name of the embedding model.
        api_key (str): API key for the Cohere API.
        device (str): Device to run the embedding model on.
        **kwargs: Additional parameters for the Cohere API.
        """

        self.cohere_client = cohere.Client(api_key=api_key)

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True},
            cache_folder=embedding_model_path
        )

        self.vdb = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embedding_model
        )

        self.top_k = top_k

        with open(sys_prompt_path) as f:
            self.sys_prompt = f.read()

        self.cohere_params = kwargs

    def _format_chat_history(self,
                             history: List[Tuple[str, str]]
                             ) -> List[Dict[str, str]]:
        """
        Format the chat history for the Cohere API.

        Parameters:
        history (List[Tuple[str, str]]): List of user and assistant messages.

        Returns:
        List[Dict[str, str]]: Formatted chat history.
        """

        chat_history = []

        for user_message, assistant_message in history:
            chat_history.append({'role': 'USER',
                                 'message': user_message})
            chat_history.append({'role': 'CHATBOT',
                                 'message': assistant_message})

        return chat_history

    def _get_response(self,
                      message: str,
                      history: List[Tuple[str, str]]
                      ) -> Generator[str, None, None]:
        """
        Get the response from the chatbot.

        Parameters:
        message (str): User message.
        history (List[Tuple[str, str]]): List of user and assistant messages.

        Yields:
        str: Output tokens from the chatbot.
        """

        documents = self.vdb.similarity_search(message, self.top_k)

        context = "\n".join([doc.page_content for doc in documents])

        message_augmented = f"{self.sys_prompt}Avec l'aide de ces extraits du\
                              code civil français: {context} réponds à la\
                              question suivante: {message}"

        chat_history = self._format_chat_history(history) if history else []

        response = self.cohere_client.chat_stream(message=message_augmented,
                                                  chat_history=chat_history,
                                                  **self.cohere_params)

        full_response = ""
        for event in response:
            if event.event_type == "text-generation":
                full_response += event.text
                yield full_response

    def deploy_chatbot(self,
                       title: str,
                       description: str) -> None:
        """
        Deploy the chatbot.

        Parameters:
        title (str): Title of the chatbot.
        description (str): Description of the chatbot.
        """

        gr.ChatInterface(fn=self._get_response,
                         title=title,
                         description=description).launch()


embedding_model_path = os.path.join(os.getcwd(), 'model')
vector_db_path = os.path.join(os.getcwd(), 'chroma-db')
system_prompt_path = os.path.join(os.getcwd(), 'system_prompt.txt')

ai_jurist = AIJurist(embedding_model_path=embedding_model_path,
                     vector_db_path=vector_db_path,
                     sys_prompt_path=system_prompt_path,
                     temperature=0.15)

title = "Votre assistant juridique"

description = "Posez une question juridique et votre assistant juridique vous\
               répondra en se basant sur le code civil français."

ai_jurist.deploy_chatbot(title=title,
                         description=description)
