from langchain_ollama import OllamaLLM
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
import os

os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"

class Advisor():
  def __init__(self):
    self.answer_template = """
    당신은 한국어를 잘하고 피부질환 관련해서 지식이 깊은 사람입니다.
    진단결과: {pred}

    위의 진단결과를 바탕으로 환자가 조치해야 되는 일들을 작성해주세요.
    만약 진단결과가 '정상'일 시에는 건강한 피부를 유지하는 방법을 작성해주세요.
    """

    self.llm = OllamaLLM(model="gemma3:4b", temperature=0)

    self.answer_prompt = ChatPromptTemplate.from_template(self.answer_template)
    self.answer_chain = LLMChain(llm=self.llm, prompt=self.answer_prompt)

  def answer(self, pred):
    response = self.answer_chain.invoke(
      input=pred
    )

    return response["text"]

