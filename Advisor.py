from langchain.llms import Ollama
import os

os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
ollama_model = Ollama(model="gemma3:4b")