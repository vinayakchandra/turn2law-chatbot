import os

import streamlit as st
from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

llm = ChatGroq(model_name="llama-3.3-70b-versatile")

systemPrompt = """
You are an advanced legal chatbot designed specifically for Indian law, equipped with comprehensive knowledge of legal principles, procedures, and statutes relevant to the Indian legal system. Your primary goal is to assist users in understanding legal concepts, providing accurate information, and guiding them through legal queries in a user-friendly manner.
Your task is to generate informative responses to user inquiries regarding Indian law. This includes answering questions about legal rights, processes, and relevant laws affecting individuals and businesses in India.

Please keep in mind the following details while responding: 

Ensure that all information provided is accurate and up-to-date according to the latest Indian laws and regulations.
Offer clear explanations and avoid using overly complex legal jargon, making the information accessible to users without a legal background.
Provide relevant examples or case references when necessary to illustrate legal concepts.

For instance, if a user asks about the process of filing a complaint in a consumer court, you might explain the steps involved, relevant laws, and any important deadlines or requirements they should be aware of.
"""

system_message = SystemMessage(content=systemPrompt)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [system_message]
if "latest_query" not in st.session_state:
    st.session_state.latest_query = ""
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# Load Hugging Face Embeddings Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Function to process uploaded PDF and store embeddings
def process_pdf(uploaded_file):
    # Save uploaded file locally
    pdf_path = os.path.join("temp_pdf.pdf")
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Store in ChromaDB
    vector_db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db")
    st.session_state.vector_db = vector_db
    st.success("✅ PDF processed and stored in ChromaDB!")

    # Delete temporary file
    os.remove(pdf_path)


# Function to retrieve relevant legal text from PDF
def get_relevant_text(query):
    if st.session_state.vector_db:
        docs = st.session_state.vector_db.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in docs])
    return ""


# Function to get legal advice with streaming response
def get_legal_advice(query):
    # Retrieve relevant text from PDF
    retrieved_text = get_relevant_text(query)

    # Append user message
    st.session_state.chat_history.append(HumanMessage(content=query))

    # Create context-aware prompt
    full_prompt = f"Context:\n{retrieved_text}\n\nUser Question: {query}\n\nAnswer:"

    # Display user input in chat format
    with st.chat_message("user"):
        st.write(query)

    # Display AI response in streaming format
    with st.chat_message("assistant"):
        response_container = st.empty()  # Placeholder for streaming text
        streamed_response = ""

        for chunk in llm.stream(st.session_state.chat_history + [HumanMessage(content=full_prompt)]):
            streamed_response += chunk.content  # Accumulate response
            response_container.write(streamed_response)  # Update displayed text

    # Append AI response to chat history
    st.session_state.chat_history.append(AIMessage(content=streamed_response))

    return streamed_response


# Streamlit UI
st.set_page_config(page_title="T2L", page_icon="⚖️")
st.title("⚖️ Turn2Law AI Law Chatbot")
# st.write("Ask any legal question, and I'll try to help using uploaded documents and general legal knowledge.")

# Upload PDF (in sidebar)
with st.sidebar:
    uploaded_file = st.file_uploader("📄 Upload PDF document", type=["pdf"])

    if uploaded_file:
        process_pdf(uploaded_file)

# Display chat history excluding system messages
for msg in st.session_state.chat_history:
    if isinstance(msg, SystemMessage):  # Skip system messages
        continue

    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

query = st.chat_input("Enter your legal question here...")

if query:
    st.session_state.latest_query = query  # Store query
    with st.spinner("Thinking..."):
        response = get_legal_advice(query)
    st.session_state.latest_query = ""  # Clear input after response
    # st.rerun()  # Refresh UI to update chat

# Clear chat button
if st.button("🗑 Clear Chat"):
    st.session_state.chat_history = [system_message]
    st.session_state.latest_query = ""
    st.session_state.vector_db = None
    st.rerun()
