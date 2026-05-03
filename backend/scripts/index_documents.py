import sys
from pathlib import Path

# This line allows the script to find the 'src' folder
sys.path.append(str(Path(__file__).parent.parent))

from src.services.document_indexer import DocumentIndexer
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    # 1. Instantiate our class
    # IMPORTANT: Ensure you created an index named 'compliance-docs' in the Pinecone dashboard!
    indexer = DocumentIndexer(index_name="AD-Compliance-DOCS")

    # 2. Define where our PDFs are
    data_dir = Path(__file__).parent.parent / "data"
    pdf_paths = [
        data_dir / "1001a-influencer-guide-508_1.pdf",
        data_dir / "youtube-ad-specs.pdf"
    ]

    # 3. Process each PDF
    all_chunks = []
    for pdf_path in pdf_paths:
        if pdf_path.exists():
            chunks = indexer.load_and_split(str(pdf_path))
            all_chunks.extend(chunks)
        else:
            logger.error(f"Could not find file: {pdf_path}")

    # 4. Upload everything at once
    if all_chunks:
        indexer.index_to_pinecone(all_chunks)
        logger.info("Pipeline finished successfully!")
    else:
        logger.warning("No documents were found to process.")

if __name__ == "__main__":
    main()