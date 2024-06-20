import os
import joblib
from llama_parse import LlamaParse
from config import LLAMAPARSE_API_KEY

def load_or_parse_data(pdf_file):
    data_file = f"data/parsed_data_{os.path.basename(pdf_file)}.pkl"
    if os.path.exists(data_file):
        # Load the parsed data from the file
        parsed_data = joblib.load(data_file)
    else:
        # Perform the parsing step and store the result in llama_parse_documents
        parsing_instruction = """The provided document is a project report of the computer vision project where the event classification of the event in cricket is attempted.
        This document contains all the tech stack details, approaches used, diagrams and many more project specific details.
        It contains many diagrams.
        Try to be precise while answering the questions"""
        parser = LlamaParse(api_key=LLAMAPARSE_API_KEY,
                            result_type="markdown",
                            parsing_instruction=parsing_instruction,
                            max_timeout=5000)
        llama_parse_documents = parser.load_data(pdf_file)

        # Save the parsed data to a file
        joblib.dump(llama_parse_documents, data_file)
        parsed_data = llama_parse_documents

    return parsed_data
