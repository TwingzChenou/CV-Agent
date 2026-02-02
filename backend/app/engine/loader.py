import sys
import os
from pathlib import Path

current_file = Path(__file__).resolve()
backend_root = current_file.parent.parent.parent
sys.path.append(str(backend_root))

from llama_index.core.node_parser import SentenceSplitter
from app.core.logging import setup_logging
from llama_index.core import SimpleDirectoryReader
from app.engine.index import run_indexing_pipeline
import logging
from llama_parse import LlamaParse

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

#load documents
def load_documents(name_doc):
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    filepath = project_root / "data" / name_doc
    return filepath


#setup splitter
def setup_splitter(filepath):
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",
        verbose=True,
        language="fr"
    )
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(input_files=[str(filepath)], file_extractor=file_extractor).load_data()
    return documents


def main():
    logger.info("🚀 Starting document loading...")

    filepath = load_documents("CV ATS.pdf")
    logger.info("✅ Document loading completed.")
    
    document_chunks = setup_splitter(filepath)
    logger.info("🚀 Starting document splitter...")

    run_indexing_pipeline(document_chunks)
    logger.info("✅ Document splitter setup completed.")

if __name__ == "__main__":
    sys.exit(main())
