from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever
import langchain

# langchain.debug = True

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()

db = Chroma(
  persist_directory="emb",
  embedding_function=embeddings,
)

retriever = RedundantFilterRetriever(
  embeddings=embeddings,
  chroma=db,
)

# retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
  llm=chat,
  retriever=retriever,
  chain_type="stuff",
)

result = chain.run(
  "What is an interesting fact about the english language?",
)

print(result)
