import streamlit as st
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI  # Import for Gemini
import speech_recognition as sr
from gtts import gTTS
import tempfile
from streamlit_chat import message

# "with" notation
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your file in Urdu")
    st.header("Urdu Voice-Enabled Document Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    # Clear Chat Button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.conversation = None
        st.write("Chat history cleared.")

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your PDF file", type=['pdf'])
        process = st.button("Process PDF")
    
    if process:
        files_text = get_files_text(uploaded_files)
        st.write("File loaded successfully...")

        # Get text chunks
        text_chunks = get_text_chunks(files_text)
        st.write("Text chunks created...")

        # Create vector store
        vectorstore = get_vectorstore(text_chunks)
        st.write("Vector Store Created...")

        # Create conversation chain using Gemini
        st.session_state.conversation = get_conversation_chain(vectorstore)
        st.session_state.processComplete = True

    if st.session_state.processComplete:
        if st.button("Record Question (Urdu)"):
            st.write("Recording...")
            audio_data = record_voice_input()

            if audio_data:
                user_question = convert_voice_to_text_urdu(audio_data)
                if user_question:
                    handle_user_input(user_question)

# Function to get the input file and read the text from it.
def get_files_text(uploaded_file):
    text = ""
    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split PDF text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Vector Store creation with HuggingFace embeddings
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base

# Create a conversation chain using the Gemini 1.5 Flash model
def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # Using ChatGoogleGenerativeAI with Gemini 1.5 Flash
    llm = ChatGoogleGenerativeAI(
        api_key = "AIzaSyA_NIjpykotxuoWsOq9D9zNpATAfN7IGyg",
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    return conversation_chain

# Function to handle the user's voice input in Urdu
def record_voice_input():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    return audio

# Convert Urdu voice input to text
def convert_voice_to_text_urdu(audio_data):
    recognizer = sr.Recognizer()
    try:
        user_input = recognizer.recognize_google(audio_data, language="ur-PK")
        st.write(f"آپ نے کہا: {user_input}")
        return user_input
    except sr.UnknownValueError:
        st.write("معذرت، میں آپ کی آواز کو نہیں سمجھ سکا۔")
    except sr.RequestError:
        st.write("نتائج حاصل نہیں کی جا سکیں، براہ کرم اپنے انٹرنیٹ کنکشن کو چیک کریں۔")
    return None

# Handle user input, send to Gemini model, and generate a response
def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})

    # Display the conversation in a chat format
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            message(chat["content"], is_user=True)
        else:
            message(chat["content"])
            convert_text_to_speech_urdu(chat["content"])

# Convert Urdu text to speech using gTTS
def convert_text_to_speech_urdu(text):
    tts = gTTS(text=text, lang='ur')
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        tts.save(f"{temp_file.name}.mp3")
        st.audio(f"{temp_file.name}.mp3")

if __name__ == '__main__':
    main()
