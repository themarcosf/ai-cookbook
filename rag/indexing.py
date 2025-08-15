###############################################################################
# Source: https://levelup.gitconnected.com/building-the-entire-rag-ecosystem-and-optimizing-every-component-8f23349b96a4
###############################################################################
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv()

import logging

import bs4

from operator import itemgetter

from langchain import hub
from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
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


# Create a text splitter to divide text into chunks with overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=30)


# Initialize the LLM
llm = ChatOpenAI(model_name="gpt-5", temperature=0)


# Prompts for RAG
prompt_from_hub = hub.pull("rlm/rag-prompt")

prompt_multiple_queries = ChatPromptTemplate.from_template("""
You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}
""")

prompt_rag_fusion = ChatPromptTemplate.from_template("""
You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):
""")

prompt_decomposition = ChatPromptTemplate.from_template("""
You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):
""")

prompt_decomposition_final = ChatPromptTemplate.from_template("""
Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the original question: {question}
""")

prompt_step_back = ChatPromptTemplate.from_messages([
    ("system", 
        "You are an expert at world knowledge. Your task is to step back and paraphrase a question "
        "to a more generic step-back question, which is easier to answer. Here are a few examples:"
    ),
    FewShotChatMessagePromptTemplate(
        example_prompt=ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")]), 
        examples=[
            {
                "input": "Could the members of The Police perform lawful arrests?",
                "output": "what can the members of The Police do?",
            },
            {
                "input": "Jan Sindel's was born in what country?",
                "output": "what is Jan Sindel's personal history?",
            },
        ]
    ),
    ("user", "{question}"),
])

prompt_step_back_final = ChatPromptTemplate.from_template("""
You are an expert of world knowledge. I am going to ask you a question. Your response should be \n
comprehensive and not contradicted with the following context if they are relevant. Otherwise, \n
ignore them if they are not relevant.

# Normal Context
{normal_context}

# Step-Back Context
{step_back_context}

# Original Question: {question}
# Answer:
""")

prompt_hyde = ChatPromptTemplate.from_template("""
Please write a scientific paper passage to answer the question
Question: {question}
Passage:
""")


# Helper functions
def format_docs(docs):
    """ Format retrieved documents """
    return "\n\n".join(doc.page_content for doc in docs)

def get_unique_union(documents: list[list]):
    """ Get unique union of retrieved documents """
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal Rank Fusion that intelligently combines multiple ranked lists """
    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # documents ranked higher (lower rank value) get a larger score
            fused_scores[doc_str] += 1 / (rank + k)

    # sort documents by their new fused scores in descending order
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return reranked_results

def format_qa_pairs(questions, answers):
        """Format Q&A pairs for decomposition"""
        formatted_string = ""
        for i, (q, a) in enumerate(zip(questions, answers), start=1):
            formatted_string += f"Question {i}: {q}\nAnswer {i}: {a}\n\n"
        return formatted_string.strip()


if __name__ == "__main__":
    logger.info(f"Loaded {len(kb)} documents to knowledge base.")
# multi-representation indexing