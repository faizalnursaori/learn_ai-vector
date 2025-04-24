from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()

with open("markdown.txt", "r") as f:
    text = f.read()

text_splitter = SemanticChunker(OpenAIEmbeddings())

documents = text_splitter.create_documents([text])

print(documents)
