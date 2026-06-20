import sys
import os
from pathlib import Path

current_file = Path(__file__).resolve()
backend_root = current_file.parent.parent.parent
sys.path.append(str(backend_root))

from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
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
def setup_splitter_pdf(filepath):
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",
        verbose=True,
        language="fr"
    )
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(input_files=[str(filepath)], file_extractor=file_extractor).load_data()
    return documents

def setup_splitter_md(filepath):
    parser = MarkdownNodeParser()
    documents = SimpleDirectoryReader(input_files=[str(filepath)]).load_data()
    nodes = parser.get_nodes_from_documents(documents)
    return nodes

def main():
    logger.info("🚀 Starting document loading...")

    cv_filepath = load_documents("Quentin_Forget_CV.pdf")
    md_filepath = load_documents("profil_quentin.md")
    logger.info("✅ Document loading completed.")
    
    document_chunks_cv = setup_splitter_pdf(cv_filepath)
    logger.info("✅ CV splitter setup completed.")

    document_chunks_md = setup_splitter_md(md_filepath)
    logger.info("✅ Profil splitter setup completed.")

    # Combine both lists of chunks to index them in a single call,
    # ensuring we can safely clean the index first without losing data.
    all_chunks = list(document_chunks_cv) + list(document_chunks_md)
    run_indexing_pipeline(all_chunks)
    logger.info("✅ Indexing of all CV and Profil documents completed.")

if __name__ == "__main__":
    sys.exit(main())
