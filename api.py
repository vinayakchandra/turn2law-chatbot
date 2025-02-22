import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
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
chat_history = [system_message]
latest_query = ""
vector_db = None

# Load Hugging Face Embeddings Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI()


# Function to process uploaded PDF and store embeddings
def process_pdf(uploaded_file):
    global vector_db
    # Save uploaded file locally
    pdf_path = os.path.join("temp_pdf.pdf")
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.file.read())

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Store in ChromaDB
    vector_db = Chroma.from_documents(chunks, embedding_model)

    # Delete temporary file
    os.remove(pdf_path)


# Function to retrieve relevant legal text from PDF
def get_relevant_text(query):
    global vector_db
    if vector_db:
        docs = vector_db.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in docs])
    return ""


# Function to get legal advice with streaming response
def get_legal_advice(query):
    global chat_history
    # Retrieve relevant text from PDF
    retrieved_text = get_relevant_text(query)

    # Append user message
    chat_history.append(HumanMessage(content=query))

    # Create context-aware prompt
    full_prompt = f"Context:\n{retrieved_text}\n\nUser Question: {query}\n\nAnswer:"

    # Display AI response in streaming format
    streamed_response = ""
    for chunk in llm.stream(chat_history + [HumanMessage(content=full_prompt)]):
        streamed_response += chunk.content  # Accumulate response

    # Append AI response to chat history
    chat_history.append(AIMessage(content=streamed_response))

    return streamed_response


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    process_pdf(file)
    return JSONResponse(content={"message": "PDF uploaded successfully"}, status_code=200)


@app.get("/get_legal_advice")
async def get_advice(query: str):
    advice = get_legal_advice(query)
    return JSONResponse(content={"advice": advice}, status_code=200)


@app.post("/clear_chat")
async def clear_chat():
    global chat_history
    global latest_query
    global vector_db
    chat_history = [system_message]
    latest_query = ""
    vector_db = None
    return JSONResponse(content={"message": "Chat cleared successfully"}, status_code=200)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
