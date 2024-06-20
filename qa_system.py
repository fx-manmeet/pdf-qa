from groq import Groq
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from config import GROQ_API_KEY

def setup_qa_system(vs, embed_model):
    chat_model = ChatGroq(temperature=0,
                          model_name="mixtral-8x7b-32768",
                          api_key=GROQ_API_KEY)
    
    vectorstore = Chroma(embedding_function=embed_model,
                         persist_directory="chroma_db_llamaparse",
                         collection_name="rag1")
    
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    custom_prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    
    qa = RetrievalQA.from_chain_type(llm=chat_model,
                                     chain_type="stuff",
                                     retriever=retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={"prompt": prompt})
    return qa
