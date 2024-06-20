from vector_db import create_vector_database
from qa_system import setup_qa_system

def query_pipeline(pdf_file, question):
    vs, embed_model = create_vector_database(pdf_file)
    qa_system = setup_qa_system(vs, embed_model)
    response = qa_system.invoke({"query": question})
    return response['result']
