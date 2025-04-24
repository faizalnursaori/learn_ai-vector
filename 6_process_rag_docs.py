import os
import json

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from mistralai import Mistral
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

openai_client = OpenAI()

chroma_client = chromadb.HttpClient(host="localhost", port=8010)

client = Mistral(api_key=MISTRAL_API_KEY)

openai_ef = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"
)

# ocr_response = client.ocr.process(
#     model="mistral-ocr-latest",
#     document={
#         "type": "document_url",
#         "document_url": "https://www.hasbro.com/common/instruct/00009.pdf",
#     },
# )

# content = ocr_response.model_dump()

# markdown_content = ""

# for page in content.get("pages", []):
#     if "markdown" in page:
#         markdown_content += page["markdown"] + "\n\n"

# text_splitter = SemanticChunker(OpenAIEmbeddings())

# documents = text_splitter.create_documents([markdown_content])

# chroma_client.delete_collection("monopoly-guide")
# collection = chroma_client.create_collection("monopoly-guide", embedding_function=openai_ef)


# collection.add(
#     documents=[doc.model_dump().get("page_content") for doc in documents],
#     ids=[str(i) for i in range(len(documents))]
# )

collection = chroma_client.get_collection("monopoly-guide", embedding_function=openai_ef)
# print(collection.count())

result = collection.query(
    query_texts=["What is player role of banker?"], n_results=5, include=["distances", "documents"]
)

# print(json.dumps(result, indent=3, sort_keys=True))

response = openai_client.chat.completions.create(
    model = "gpt-4o",
    messages=[
        {
            "role": "system",
            "content": f"You have to answer the question based on the provided context only! The context is: {result.get('documents')}"
        },
        {
            "role": "user",
            "content": "What is player role of banker?"
        }
    ]
)

content = response.choices[0].message.content
print(content)