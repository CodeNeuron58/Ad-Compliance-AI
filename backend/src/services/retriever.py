from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pydantic import BaseModel, Field
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 1. We define a Pydantic schema for the tool's input.
# Why? This forces the LLM to format its search query correctly as a JSON object,
# preventing errors where the LLM passes weird arguments to our Python function.
class ComplianceSearchInput(BaseModel):
    query: str = Field(description="The specific compliance question or keyword to search for.")

# 2. We use the @tool decorator. 
# Why? This tells LangChain "Take this standard Python function and convert it 
# into a JSON schema that OpenAI's API can understand."
@tool("search_compliance_rules", args_schema=ComplianceSearchInput)
def search_compliance_rules(query: str) -> str:
    """
    Search the vector database for official compliance rules, FTC guidelines, 
    and YouTube advertising specifications. 
    Use this tool whenever you need to verify if a script or video idea is legal and compliant.
    """
    # THE DOCSTRING ABOVE IS CRITICAL! 
    # The LLM literally reads that docstring to decide WHEN to use the tool. 
    # If the docstring is bad, the AI won't use the tool.
    
    logger.info(f"LLM decided to search for: '{query}'")
    
    try:
        # Initialize embeddings and connect to Pinecone (same as ingestion)
        embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
        vector_store = PineconeVectorStore(
            index_name="compliance-docs",
            embedding=embeddings
        )

        # Perform a similarity search. 'k=3' means we fetch the top 3 most relevant chunks.
        # Why k=3? LLMs have a "context window" limit. If we return 50 chunks,
        # it costs too much money and the LLM gets confused (the "Lost in the Middle" phenomenon).
        docs = vector_store.similarity_search(query, k=3)

        if not docs:
            return "No relevant compliance rules found for this query."

        # Format the results into a single string for the LLM to read.
        # Why? LLMs read text strings, not Python list objects.
        # We separate chunks with "---" so the LLM can distinguish different rules.
        formatted_results = "\n\n---\n\n".join(
            [f"Source: {doc.metadata.get('source', 'Unknown')}\nRule: {doc.page_content}" for doc in docs]
        )

        return formatted_results
    except Exception as e:
        logger.error(f"Error during vector search: {str(e)}")
        # We return the error as a string so the LLM knows the tool failed
        # and can tell the user, rather than crashing the whole application.
        return f"Error occurred while searching the database: {str(e)}"