###############################################################################
# Source: https://levelup.gitconnected.com/building-the-entire-rag-ecosystem-and-optimizing-every-component-8f23349b96a4
###############################################################################
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv()

import logging
import pprint

from datetime import date
from typing import Optional

from langchain.prompts import ChatPromptTemplate
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
class TutorialSearch(BaseModel):
    """ Search over a database of tutorial videos. """

    content_search: str = Field(..., description="Similarity search query applied to video transcripts.")
    title_search: str = Field(..., description="Alternate version of the content search query to apply to video titles.")
    min_view_count: Optional[int] = Field(None, description="Minimum view count filter, inclusive.")
    max_view_count: Optional[int] = Field(None, description="Maximum view count filter, exclusive.")
    earliest_publish_date: Optional[date] = Field(None, description="Earliest publish date filter, inclusive.")
    latest_publish_date: Optional[date] = Field(None, description="Latest publish date filter, exclusive.")
    min_length_sec: Optional[int] = Field(None, description="Minimum video length in seconds, inclusive.")
    max_length_sec: Optional[int] = Field(None, description="Maximum video length in seconds, exclusive.")

    def pretty_print(self) -> None:
        """ Print the populated fields of the model. """
        for field in TutorialSearch.model_fields:
            if getattr(self, field) is not None:
                logger.info(f"{field}: {getattr(self, field)}")


# Initialize the embedding model
embeddings = OpenAIEmbeddings()


# Initialize the LLM
llm = ChatOpenAI(model_name="gpt-5", temperature=0).with_structured_output(TutorialSearch)


# Expert personas
persona_query_analyzer = """
You are an expert at converting user questions into database queries. You have access to a database of 
tutorial videos about a software library for building LLM-powered applications. Given a question, return 
a database query optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them.
"""


# Prompts
prompt_query_analyzer = ChatPromptTemplate.from_messages([
    ("system", persona_query_analyzer), 
    ("human", "{question}")
])


if __name__ == "__main__":
    logger.info("1st method: Query structuring")

    doc = { 
      'source': 'pbAd8O1Lvm4',
      'title': 'Self-reflective RAG with LangGraph: Self-RAG and CRAG',
      'description': 'Unknown',
      'view_count': 11922,
      'thumbnail_url': 'https://i.ytimg.com/vi/pbAd8O1Lvm4/hq720.jpg',
      'publish_date': '2024-02-07 00:00:00',
      'length': 1058,
      'author': 'LangChain'
    }

    logger.info("Example database document:\n{\n%s\n}", pprint.pformat(doc, indent=2)[1:-1])

    query_structuring = prompt_query_analyzer | llm

    query = "rag from scratch"
    logger.info(f"Sample query: {query}")

    query_structuring.invoke({"question": query}).pretty_print()

    query = "videos on chat langchain published in 2023"
    logger.info("")
    logger.info(f"Sample query: {query}")

    query_structuring.invoke({"question": query}).pretty_print()

    query = "how to use multi-modal models in an agent, only videos under 5 minutes"
    logger.info("")
    logger.info(f"Sample query: {query}")

    query_structuring.invoke({"question": query}).pretty_print()
