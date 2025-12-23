"""script for running fact file through langchain"""

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
  separator="\n",
  chunk_size=200,
  chunk_overlap=100,
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(
  text_splitter,
)

db = Chroma.from_documents(
  docs,
  embedding=embeddings,
  persist_directory="emb",
)

results = db.similarity_search_with_score(
  "What is an interesting fact about the english language?",
  k=1,
)

# for doc in docs:
#   print(doc.page_content)
#   print("\n")

for result in results:
  print("\n")
  print(result[1])
  print(result[0].page_content)
