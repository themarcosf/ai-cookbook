###############################################################################
# Source: https://levelup.gitconnected.com/building-the-entire-rag-ecosystem-and-optimizing-every-component-8f23349b96a4
###############################################################################
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv()

import logging

from typing import Literal

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.utils.math import cosine_similarity
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress httpx debug/info logs
logging.getLogger("httpx").setLevel(logging.WARNING)


###############################################################################
# Implementation
###############################################################################

# Data models
class RouteQuery(BaseModel):
    """ Route a user query to the most relevant datasource. """

    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="Given a user question, choose which datasource would be most relevant for answering their question.",
    )


# Initialize the embedding model
embeddings = OpenAIEmbeddings()


# Initialize the LLM
llm = ChatOpenAI(model_name="gpt-5", temperature=0).with_structured_output(RouteQuery)


# Expert personas
persona_router = """
You are an expert at routing a user question to the appropriate data source.

Based on the programming language the question is referring to, route it to the relevant data source.
"""

persona_physics = """
You are a very smart physics professor. You are great at answering questions about physics in a concise 
and easy to understand manner. When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}
"""

persona_math = """
You are a very good mathematician. You are great at answering math questions.
You are so good because you are able to break down hard problems into their component parts,
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{query}
"""


# Prompts
prompt_router = ChatPromptTemplate.from_messages([
    ("system", persona_router),
    ("human", "{question}"),
])


# Helper functions
def choose_route(result):
    """ Determine downstream logic based on router's output. """
    mapping = {
        "python_docs": "chain for python_docs",
        "js_docs": "chain for js_docs",
        "golang_docs": "chain for golang_docs",
    }

    return mapping.get(result.datasource.lower())

def semantic_routing(input):
    """ Route input query to most similar prompt template."""

    persona_templates = [persona_physics, persona_math]
    persona_embeddings = embeddings.embed_documents(persona_templates)
    
    query_embedding = embeddings.embed_query(input["query"])

    similarity = cosine_similarity([query_embedding], persona_embeddings)[0]

    most_similar_index = similarity.argmax()

    chosen_prompt = persona_templates[most_similar_index]

    logger.info(f"Using {'MATH' if most_similar_index == 1 else 'PHYSICS'} template.")
    
    return PromptTemplate.from_template(chosen_prompt)


if __name__ == "__main__":
    logger.info("1st method: Logical Routing")

    question = """
    Why doesn't the following code work:

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
    prompt.invoke("french")
    """
    logger.info(f"Question: {question}")

    logical_router = prompt_router | llm

    result = logical_router.invoke({"question": question})
    logger.info(f"Logical router output: {result}")

    switchboard_chain = logical_router | RunnableLambda(choose_route)

    datasource = switchboard_chain.invoke({"question": question})
    logger.info(f"Switchboard output: {datasource}")

    logger.info("")
    logger.info("2nd method: Semantic Routing")

    question = "What's a black hole"
    logger.info(f"Question: {question}")

    semantic_router = (
        {"query": RunnablePassthrough()}
        | RunnableLambda(semantic_routing)
    )

    semantic_router.invoke(question)