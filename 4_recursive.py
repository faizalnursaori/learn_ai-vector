from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("markdown.txt", "r") as f:
    text = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(text)

for chunk in chunks:
    print(chunk)
