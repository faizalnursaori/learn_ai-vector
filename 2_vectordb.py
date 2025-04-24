from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

import os
import chromadb
import json

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

load_dotenv()

client = chromadb.HttpClient("localhost", 8010)

openai_ef = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

fruit_strings = [
    "Bananas are yellow and curved.",
    "Apples grow on trees in orchards.",
    "Oranges have a thick peel that protects the fruit.",
    "Banana smoothies are delicious and nutritious.",
    "Red apples are sweet, while green apples are often tart.",
    "Orange juice is high in vitamin C.",
    "Bananas are rich in potassium.",
    "Apple pie is a classic American dessert.",
    "Orange zest adds flavor to many recipes.",
    "Bananas ripen from green to yellow to brown.",
    "An apple a day keeps the doctor away.",
    "The pith of an orange is the white part under the peel.",
    "Bananas grow in clusters called hands.",
    "Apple cider is a popular fall drink.",
    "Oranges originated in Southeast Asia.",
    "Banana bread is a good way to use overripe bananas.",
    "The skin of an apple contains many nutrients.",
    "Valencia oranges are known for juicing.",
    "Bananas are technically berries, botanically speaking.",
    "Granny Smith is a popular variety of green apple."
]

collection = client.get_collection("fruits", embedding_function=openai_ef)
# print(collection.count())

result = collection.query(
    query_texts=["Is banana yellow?"],
    n_results=3,
    include=["distances", "documents"]
)

print(json.dumps(result, indent=3, sort_keys=True))

# collection = client.create_collection("fruits", embedding_function=openai_ef)

# collection.add(
#     documents=fruit_strings,
#     ids=[str(i) for i in range(len(fruit_strings))]
# )