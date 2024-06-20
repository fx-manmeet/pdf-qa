import os
import joblib
from llama_parse import LlamaParse
from config import LLAMAPARSE_API_KEY

def load_or_parse_data():
    data_file = "data/parsed_data.pkl"
    if os.path.exists(data_file):
        # Load the parsed data from the file
        parsed_data = joblib.load(data_file)
    else:
        # Perform the parsing step and store the result in llama_parse_documents
        parsing_instruction = """
        Try to be precise while answering the questions"""
        parser = LlamaParse(api_key=LLAMAPARSE_API_KEY,
                            result_type="markdown",
                            parsing_instruction=parsing_instruction,
                            max_timeout=5000)
        llama_parse_documents = parser.load_data("data\\Internship_report_456.pdf")

        # Save the parsed data to a file
        joblib.dump(llama_parse_documents, data_file)
        parsed_data = llama_parse_documents

    return parsed_data
