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

# Initialize a web document loader with specific parsing instructions
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={ "parse_only": bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")) },
)


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
    # Ask a question using RAG chain
    question = "What is Task Decomposition for LLM agents?"
    logger.info(f"Question: {question}")
    logger.info("")

    # Load filtered content from web page into documents
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} documents from web page.")
    for doc in docs:
        logger.info(f"Document content length: {len(doc.page_content)} characters.")

    # Split loaded documents into smaller chunks
    splits = text_splitter.split_documents(docs)
    logger.info(f"Split document into {len(splits)} chunks.")

    # Embed text chunks and store them in a Chroma vector store for similarity search
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    logger.info(f"Created vector store with {len(vectorstore)} embedded chunks.")

    # Create a retriever from vector store
    retriever = vectorstore.as_retriever()
    logger.info("Retriever created from vector store.")

    # Define basic RAG chain
    basic_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_from_hub
        | llm
        | StrOutputParser()
    )

    response = basic_chain.invoke(question)
    logger.info(f"Response: {response}")

    logger.info("")
    logger.info("1st TECHNIQUE: multi-query generation")

    generate_queries = (
        prompt_multiple_queries 
        | llm
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    query_chain = generate_queries | retriever.map() | get_unique_union

    multi_query_chain = (
        {"context": query_chain, "question": itemgetter("question")} 
        | prompt_from_hub
        | llm
        | StrOutputParser()
    )

    response = multi_query_chain.invoke({"question": question})
    logger.info(f"Response: {response}")
    
    logger.info("")
    logger.info("2nd TECHNIQUE: RAG-fusion [RRF - Reciprocal Rank Fusion]")

    generate_queries = (
        prompt_rag_fusion 
        | llm
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    query_chain = generate_queries | retriever.map() | reciprocal_rank_fusion

    rag_fusion_chain = (
        {"context": query_chain, "question": itemgetter("question")} 
        | prompt_from_hub
        | llm
        | StrOutputParser()
    )

    response = rag_fusion_chain.invoke({"question": question})
    logger.info(f"Response: {response}")

    logger.info("")
    logger.info("3rd TECHNIQUE: Decomposition")

    decomp_question = "What are the main components of an LLM-powered autonomous agent system?"
    logger.info(f"Question: {decomp_question}")

    generate_queries = (
        prompt_decomposition 
        | llm
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    sub_questions = generate_queries.invoke({"question": decomp_question})
    logger.info(f"Sub-questions: {sub_questions}")

    sub_answers = []
    for sq in sub_questions:
        docs = retriever.invoke(sq)
        answer = (prompt_from_hub | llm | StrOutputParser()).invoke({"context": docs, "question": sq})
        sub_answers.append(answer)

    context = format_qa_pairs(sub_questions, sub_answers)

    decomposition_chain = (
        prompt_decomposition_final
        | llm
        | StrOutputParser()
    )

    response = decomposition_chain.invoke({"context": context, "question": decomp_question})
    logger.info(f"Response: {response}")

    logger.info("")
    logger.info("4th TECHNIQUE: Step-back prompting")

    generate_step_back = prompt_step_back | llm | StrOutputParser()
    step_back_question = generate_step_back.invoke({"question": question})

    logger.info(f"Original Question: {question}")
    logger.info(f"Step-Back Question: {step_back_question}")

    step_back_chain = (
        {
            "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
            "step_back_context": generate_step_back | retriever,
            "question": lambda x: x["question"],
        }
        | prompt_step_back_final
        | llm
        | StrOutputParser()
    )

    response = step_back_chain.invoke({"question": question})
    logger.info(f"Response: {response}")

    logger.info("")
    logger.info("5th TECHNIQUE: HyDE - Hypothetical Document Embeddings")

    generate_hydoc = (
        prompt_hyde 
        | llm
        | StrOutputParser() 
    )

    hydoc = generate_hydoc.invoke({"question": question})
    retrieved_docs = (generate_hydoc | retriever).invoke({"question": question})
    
    response = (prompt_from_hub | llm | StrOutputParser()).invoke({"context": retrieved_docs, "question": question})
    logger.info(f"Response: {response}")

