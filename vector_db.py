import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from data_loader import load_or_parse_data

def create_vector_database(pdf_file):
    llama_parse_documents = load_or_parse_data(pdf_file)
    
    output_md = f"data/output_{os.path.basename(pdf_file)}.md"
    with open(output_md, 'w') as f:  # Open the file in write mode ('w')
        for doc in llama_parse_documents:
            f.write(doc.text + '\n')

    loader = DirectoryLoader('data/', glob=f"**/{os.path.basename(output_md)}", loader_cls=TextLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    vs = Chroma.from_documents(
        documents=documents,
        embedding=embed_model,
        persist_directory=f"chroma_db_{os.path.basename(pdf_file)}",  # Use PDF file name in directory
        collection_name=f"pdf_specific_collection{os.path.basename(pdf_file)}"  # Adjust collection name as needed
    )
    return vs, embed_model
