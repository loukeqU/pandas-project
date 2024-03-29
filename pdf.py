import os
from pathlib import Path
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
# from llama_index.readers import PDFReader
from llama_index.core import SimpleDirectoryReader


def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )

    return index

pdf_path = os.path.join("data", "United_States.pdf")
# us_pdf = PDFReader().load_data(file=pdf_path)
us_pdf = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
us_index = get_index(us_pdf, "united states")
us_engine = us_index.as_query_engine()