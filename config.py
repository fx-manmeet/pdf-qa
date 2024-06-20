import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

LLAMAPARSE_API_KEY = os.getenv('LLAMA_CLOUD_API_KEY')
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
