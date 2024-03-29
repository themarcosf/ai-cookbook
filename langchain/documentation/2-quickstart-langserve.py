# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# 1. Load Retriever dependencies
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 2. Create Tools dependencies
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults

# 3. Create Agent dependencies
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents import create_openai_functions_agent

# 4. App definition dependencies
from fastapi import FastAPI

# 5. Adding chain route dependencies
from typing import List
from langserve import add_routes
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage


# 1. Load Retriever
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
page = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(page)
embeddings = OpenAIEmbeddings()
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

# 2. Create Tools
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)
search_tool = TavilySearchResults()
tools = [retriever_tool, search_tool]

# 3. Create Agent
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route: localhost:8000/agent/playground
class Input(BaseModel):
    input: str
    chat_history: List[BaseMessage] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "location"}},
    )

class Output(BaseModel):
    output: str

add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output),
    path="/agent",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)