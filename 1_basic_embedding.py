from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

response = client.embeddings.create(
    input="Apple is a company in San Francisco",
    model="text-embedding-3-small"
)

print(response.data[0].model_dump_json())