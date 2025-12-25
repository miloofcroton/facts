from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
# from langchain_openapi import ChatOpenAI

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()

db = Chroma(
  persist_directory="emb",
  embedding_function=embeddings,
)

retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
  llm=chat,
  retriever=retriever,
  chain_type="stuff",
)

result = chain.run(
  "What is an interesting fact about the english language?",
)

print(result)
