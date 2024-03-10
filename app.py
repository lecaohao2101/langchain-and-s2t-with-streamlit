import os
import dotenv
import openai
import streamlit as st
from dotenv import load_dotenv
from googletrans import Translator
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import docx2txt

# import API key from .env file
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# speech to text
def transcribe(input_file):
    transcript = openai.Audio.transcribe("whisper-1", input_file)  # dictionary
    return transcript

# save file upload
def save_file(audio_bytes, file_name):
    with open(file_name, "wb") as f:
        f.write(audio_bytes)

# read file upload and transcribe
def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = transcribe(audio_file)

    return transcript["text"]

# change language
def translate_text(text, language_desire):
    translator = Translator()
    translated_text = translator.translate(text, dest=language_desire)
    return translated_text.text

def transcribe_and_translate(input_file, language_desire):
    # get name_file_input and remove extension
    name_file_input = os.path.splitext(input_file.name)[0]

    # render type input_file /mp4
    audio_file_name = f"{name_file_input}.{input_file.type.split('/')[1]}" #

    # create .txt
    transcript_file_name = f"{name_file_input}_transcript.txt"

    # save file
    save_file(input_file.read(), audio_file_name)

    # contains text of audio_file_name
    transcript_text = transcribe_audio(audio_file_name)

    # Define the folder to save the transcript file
    knowledgebase_folder = "knowledgebase"

    # Check if the folder exists, if not, create it
    if not os.path.exists(knowledgebase_folder):
        os.makedirs(knowledgebase_folder)

    # Specify the path to save the transcript file
    transcript_file_path = os.path.join(knowledgebase_folder, transcript_file_name)

    # Save transcript file in the knowledgebase folder
    with open(transcript_file_path, "w") as f:
        f.write(transcript_text)

    st.header("Language Of The File")
    st.write(transcript_text)

    # ... (remaining code, unchanged)

    if language_desire.lower() != input_file.type.split('-')[0].lower():
        # Translate only if the selected language is different from the input language
        translated_text = translate_text(transcript_text, language_desire)
        translated_file_name = f"{name_file_input}_transcript_{language_desire}.txt"

        with open(translated_file_name, "w") as f:
            f.write(translated_text)

        st.header(f"Translation For File Language ({language_desire})")
        st.write(translated_text)
        st.download_button(f"Download Translation For File Language ({language_desire})", translated_text,
                           file_name=translated_file_name)

    st.download_button("Download Language Of The File", transcript_text, file_name=transcript_file_name)

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your Documents")
    st.header("Ask your Documents 💬")

    # upload files
    uploaded_files = st.file_uploader("Upload your files", type=["mp4"], accept_multiple_files=True)

    # transcribe and translate uploaded files
    for uploaded_file in uploaded_files:
        transcribe_and_translate(uploaded_file, "en")  # replace "en" with your desired language

    # extract text from transcript files in the knowledgebase folder
    all_text = ""
    knowledgebase_folder = "knowledgebase"
    for file_name in os.listdir(knowledgebase_folder):
        if file_name.endswith(".txt"):
            with open(os.path.join(knowledgebase_folder, file_name), "r") as f:
                text = f.read()
            all_text += text

    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(all_text)

    # create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(texts=chunks, embedding=embeddings)  # Update the parameter names

    # show user input
    user_question = st.text_input("Ask a question about the uploaded documents:")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)

        llm = OpenAI(model_name="gpt-3.5-turbo")
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            print(response)

        st.write(response)

if __name__ == "__main__":
    main()

