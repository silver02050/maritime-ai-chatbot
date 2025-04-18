import os
from langchain_community.chat_models import ChatCohere
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai
from .maritime_bert import MaritimeBERT


class LLM:
    def __init__(self, base_model, api_key=None, model_path=None) -> None:
        self.base_model = base_model

        if base_model == "Gemini-Pro":
            os.environ["GOOGLE_API_KEY"] = api_key
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            self.model = genai.GenerativeModel("gemini-pro")
        elif base_model == "Cohere":
            os.environ["COHERE_API_KEY"] = api_key
            self.model = ChatCohere()
        elif base_model == "MaritimeBERT":
            self.model = MaritimeBERT(model_path=model_path)
        else:
            raise ValueError(f"Unsupported model: {base_model}")

    def inference(self, prompt, context=None):
        if self.base_model == "Gemini-Pro":
            response = self.model.generate_content(prompt).text
            confidence = 1.0
        elif self.base_model == "Cohere":
            chain = self.model | StrOutputParser()
            response = chain.invoke(prompt)
            confidence = 1.0
        elif self.base_model == "MaritimeBERT":
            response, confidence = self.model.generate_response(prompt, context)

        return response, confidence
