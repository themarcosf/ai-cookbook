import os
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader
from langchain.vectorstores.pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

import pinecone

# Initialize Pinecone
pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT"),
)


if __name__ == "__main__":
    print("Hello VectorStore!")

    # Load environment variables
    load_dotenv()

    loader = TextLoader(f"{os.getcwd()}/data/medium/medium.txt")
    documents = loader.load()

    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=0)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    docsearch = Pinecone.from_documents(chunks, embeddings, index_name="vector-store")
