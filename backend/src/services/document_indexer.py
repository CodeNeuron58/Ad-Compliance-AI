import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DocumentIndexer:
    def __init__(self,index_name: str):
        """ Initializes the models and connections needs for indexing documents"""
        
        # Set index name
        self.index_name = index_name
        
        # Initialize HuggingFace Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize vector store
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings
        )
        
        logger.info(f"Initialized DocumentIndexer for Pinecone index: {self.index_name}")
    
    def load_and_split(self, file_path: str) -> list:
        """Loads a single PDF and splits it into smaller chunks."""
        
        logger.info(f"Loading document: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        logger.info("Splitting document into chunks...")
        chunks = self.text_splitter.split_documents(docs)
        
        logger.info(f"Created {len(chunks)} chunks.")
        
        return chunks
    
    def index_to_pinecone(self, chunks: list):
        """Uploads the chunks to Pinecone."""
        
        logger.info(f"Uploading {len(chunks)} chunks to Pinecone...")
        
        # This automatically connects to Pinecone and uploads the vectors
        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            index_name=self.index_name
        )
        logger.info("Upload to Pinecone complete!")

        