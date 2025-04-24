from mistralai import Mistral
from dotenv import load_dotenv

import os

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

client = Mistral(
    api_key=MISTRAL_API_KEY
)

ocr_response = client.ocr.process(
    model= "mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": "https://www.hasbro.com/common/instruct/00009.pdf"
    }

)

content = ocr_response.model_dump()

markdown_content = ""

for page in content.get("pages", []):
    if "markdown" in page:
        markdown_content += page["markdown"] + "\n\n"

print(markdown_content)