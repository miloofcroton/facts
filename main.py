"""script for running fact file through langchain"""

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
  separator="\n",
  chunk_size=200,
  chunk_overlap=100,
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(
  text_splitter,
)

for doc in docs:
  print(doc.page_content)
  print("\n")
