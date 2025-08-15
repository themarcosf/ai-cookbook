###############################################################################
# Source: https://levelup.gitconnected.com/building-the-entire-rag-ecosystem-and-optimizing-every-component-8f23349b96a4
###############################################################################
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv()

import logging

from uuid import uuid4

from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings



# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress httpx debug/info logs
logging.getLogger("httpx").setLevel(logging.WARNING)

###############################################################################
# Implementation
###############################################################################

# Load blog posts to knowledge base
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
kb = loader.load()

loader = WebBaseLoader("https://lilianweng.github.io/posts/2024-02-05-human-data-quality/")
kb.extend(loader.load())


# Generate unique IDs for each original document
doc_ids = [str(uuid4()) for _ in kb]


# Initialize the LLM
llm = ChatOpenAI(model_name="gpt-5", temperature=0)


# Vectorstore to index summary embeddings
vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())


# Storage layer for parent documents
store = InMemoryByteStore()


# Link summaries to parent documents
id_key = "doc_id"


# Orchestrate vector and document retrieval 
retriever = MultiVectorRetriever(vectorstore=vectorstore, byte_store=store, id_key=id_key)


if __name__ == "__main__":
    logger.info(f"Loaded {len(kb)} documents to knowledge base.")

    logger.info("")
    logger.info("1st technique: Multi-representation indexing")
    
    summary_chain = (
        {"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
        | llm
        | StrOutputParser()
    )

    summaries = summary_chain.batch(kb, {"max_concurrency": 5})
    logger.info(f"Generated {len(summaries)} summaries.")

    summary_docs = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(summaries)]

    retriever.vectorstore.add_documents(summary_docs)

    retriever.docstore.mset(list(zip(doc_ids, kb)))