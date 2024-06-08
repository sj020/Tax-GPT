import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from operator import itemgetter
from streamlit_chat import message
from langchain.load import dumps, loads
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Hugging Face API Key
HUGGINGFACE_API_KEY = "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"

# Langsmith for monitoring and tracking of LLM
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = "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"
os.environ['TOKENIZERS_PARALLELISM'] = "false"

# Getting unique documents after doing multi-Query retrieval
def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

# Driver for Multi Query Retriever
def multi_retriever(vector_store):
    # Creating embeddings for the uploaded documents
    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""

    prompt_prespectives = ChatPromptTemplate.from_template(template)
    llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                              task="text-generation",
                              do_sample=False,huggingfacehub_api_token = HUGGINGFACE_API_KEY, temperature= 0.7)
    generate_queries = (prompt_prespectives | llm | StrOutputParser() | (lambda x: x.split("\n")))
    retrieval_chain = generate_queries | vector_store.as_retriever().map() | get_unique_union

    return retrieval_chain

# Driver for initializing the session state
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about your Tax Documents"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

# Returning the result from LLM
def conversation_chat(query, chain, history):
    result = chain.invoke({"question": query, "chat_history": history})
    history.append((query, result))
    return result

# Driver for displaying the chat
def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

# Driver for Coversational Chain
def create_conversational_chain(multi_retrieval_chain):
    load_dotenv()
    template = """
    Given the following context:
    {context}

    Answer the following question:
    {question}"""
    llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3", task="text-generation",temperature=0.75,top_p=0.95,verbose=True,streaming=True,
                              do_sample=False, huggingfacehub_api_token = HUGGINGFACE_API_KEY)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    chain = (
    {
        "context" : multi_retrieval_chain,
        "question":itemgetter("question")
    }
    | prompt
    | llm
    | StrOutputParser()
    )
    return chain

def main():
    load_dotenv()
    # Initialize session state
    initialize_session_state()
    st.title("Tax Guidance Chatbot 🤖")
    # Initialize Streamlit
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".docx" or file_extension == ".doc":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
        text_chunks = text_splitter.split_documents(text)
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                           model_kwargs={'device': 'cpu'})

        # Create vector store
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        multi_retrieval_chain = multi_retriever(vector_store)

        # Create the chain object
        chain = create_conversational_chain(multi_retrieval_chain)
        
        display_chat_history(chain)

if __name__ == "__main__":
    main()
