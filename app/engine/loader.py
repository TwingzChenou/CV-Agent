from llama_index.core.node_parser import SentenceSplitter
from app.core.logging import setup_logging
from llama_index.core import SimpleDirectoryReader
from app.engine.index import run_indexing_pipeline
import sys
import logging
from pathlib import Path

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

#load documents
def load_documents(name_doc):
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    filepath = project_root / "data" / name_doc
    reader = SimpleDirectoryReader(input_files=[str(filepath)])
    return reader.load_data()


#setup splitter
def setup_splitter(documents):
    parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    return parser.get_nodes_from_documents(documents)


def main():
    logger.info("🚀 Starting document loading...")

    documents = load_documents("CV_Quentin_Forget.pdf")
    logger.info("✅ Document loading completed.")
    
    document_chunks = setup_splitter(documents)
    logger.info("🚀 Starting document splitter...")

    run_indexing_pipeline(document_chunks)
    logger.info("✅ Document splitter setup completed.")

if __name__ == "__main__":
    sys.exit(main())
