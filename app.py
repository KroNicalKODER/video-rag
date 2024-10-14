import streamlit as st
import os
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.embeddings import OllamaEmbeddings
from moviepy.video.io.VideoFileClip import VideoFileClip
import re
import json
import shutil
from dotenv import load_dotenv
load_dotenv()

st.title("VIDEO RAG AND AI-SPLITTER")

embeddings = OllamaEmbeddings(model="llama3.2")
vectorstore_path = 'db'
vectorstore_folders = [f for f in os.listdir(vectorstore_path) if os.path.isdir(os.path.join(vectorstore_path, f))]
selected_db = st.selectbox("Select Database", vectorstore_folders)
store = {}

def change_retriever(selected_db):
  
    vectorstore = Chroma(
        persist_directory='db/'+selected_db,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever()
    return retriever

retriever = change_retriever(selected_db)

## *** FOR TEXT BASED RAG ****##
contextualize_q_system_prompt_t=(
    "Given a text and the latest user question"
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt_t = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt_t),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

store_t = {}

groq_api_key = os.getenv("groq_api_key")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

history_aware_retriever_t=create_history_aware_retriever(llm,retriever,contextualize_q_prompt_t)

system_prompt_t = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
qa_prompt_t = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_t),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

def get_session_history_t(session_id):
    if session_id not in store_t:
        store_t[session_id] = ChatMessageHistory()
    return store_t[session_id]

question_answer_chain_t=create_stuff_documents_chain(llm,qa_prompt_t)
rag_chain_t=create_retrieval_chain(history_aware_retriever_t,question_answer_chain_t)
conversational_rag_chain_t=RunnableWithMessageHistory(
            rag_chain_t,get_session_history_t,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )


if 'selected_db' not in st.session_state:
    st.session_state.selected_db = selected_db
    st.session_state.retriever = change_retriever(selected_db)

# Update the retriever if the selected db changes
if selected_db != st.session_state.selected_db:
    st.session_state.selected_db = selected_db
    st.session_state.retriever = change_retriever(selected_db)
    st.write(f"Retriever updated for database: {selected_db}")

# Access the current retriever
retriever = st.session_state.retriever

modes = ['chat','split']
selected_mode = st.selectbox("Select Mode", modes)

if 'selected_mode' not in st.session_state:
    st.session_state.selected_mode = selected_mode

if(selected_mode != st.session_state.selected_mode):
    st.session_state.selected_mode = selected_mode

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def show_clipped_videos():
    video_path = 'videos/cut'

    # List all videos in the directory
    videos = os.listdir(video_path)

    # Filter for MP4 or other supported formats if needed
    videos = [video for video in videos if video.endswith(".mp4")]

    # Display videos as thumbnails and make them playable
    for video in videos:
        st.write(f"### {video}")  # Video title or filename

        # Play the video using Streamlit's built-in st.video
        video_file_path = os.path.join(video_path, video)
        
        # Display the video with st.video
        st.video(video_file_path)

def delete_all_files_in_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # Check if it's a file and delete it
                if os.path.isfile(file_path):
                    os.remove(file_path)
                # If it's a directory, remove the directory and its contents
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        print(f"The folder {folder_path} does not exist.")

def preprocessing_split():
    contextualize_q_system_prompt=(
    "Given a transcript of a youtube podcast video and the latest user question"
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    store = {}

    history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

    system_prompt = (
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, say that you "
                    "don't know."
                    "\n\n"
                    "{context}"
                )
    qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    def get_session_history(session_id):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
    conversational_rag_chain=RunnableWithMessageHistory(
                rag_chain,get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
    return conversational_rag_chain

conversational_rag_chain = preprocessing_split()

def clean_word(word):
    # Remove punctuation, extra spaces, and convert to lowercase
    return re.sub(r'[^\w]', '', word).strip().lower()

def find_best_answer_time_range(answer, word_data):
    answer_words = answer.split()
    answer_length = len(answer_words)
    print(answer_length)
    
    best_match_score = 0  # Track the highest match score
    best_start_time = None
    best_end_time = None

    # Iterate over all possible starting points in the word data
    for i in range(len(word_data)):
        current_match_score = 0  # Reset match score for each new start position
        match_start_time = None
        match_end_time = None

        if(clean_word(word_data[i]['word']) == clean_word(answer_words[0])):
            # Loop over the words in the answer
            for j in range(answer_length):
                if i + j >= len(word_data):  # If we reach the end of word_data, break
                    break
                
                # Clean and compare the words
                if clean_word(word_data[i + j]['word']) == clean_word(answer_words[j]):
                    current_match_score += 1  # Increment match score
                    
                    # Set the start time if it's the first match
                    if match_start_time is None:
                        match_start_time = word_data[i]['start']
                if i + j < len(word_data):
                    match_end_time = word_data[i + j]['end']  # Update end time as we match words

            # Check if the current match is the best one
            if current_match_score > best_match_score:
                best_match_score = current_match_score
                best_start_time = match_start_time
                best_end_time = match_end_time

    # Return the time range of the best match and the match score percentage
    return best_start_time, best_end_time, best_match_score / answer_length if best_match_score > 0 else 0

def sanitize_filename(filename):
    return re.sub(r'[^\w\s-]', '', filename).strip().replace(' ', '_')

def clip_video(video_path, start_time, end_time, question):
    sanitized_question = sanitize_filename(question)
    output_path = f"videos/cut/{sanitized_question}.mp4"
    
    with VideoFileClip(video_path) as video:
        clipped_video = video.subclip(start_time, end_time)
        clipped_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
        print(f"Saved clip as {output_path}")

def process_text(text):
    # Split the text into chunks based on common patterns: Title, Question, Answer
    chunks = text.split("**Title:")
    
    qa_list = []

    for chunk in chunks:
        if not chunk.strip():  # Skip empty chunks
            continue

        # Initialize default values for title, question, and answer
        title = "Untitled"
        question = ""
        answer = ""

        # Split chunk into smaller sections based on "Question:" and "Answer:"
        if "**Question:" in chunk and "**Answer:**" in chunk:
            # Extract Title if it exists
            if chunk.split("**Question:")[0].strip():
                title = chunk.split("**Question:")[0].strip()

            # Extract Question and Answer parts
            question_answer_part = chunk.split("**Question:")[1]
            question_part = question_answer_part.split("**Answer:**")[0].strip()
            answer_part = question_answer_part.split("**Answer:**")[1].strip()

            question = question_part
            answer = answer_part

            # Add extracted information to the qa_list
            qa_list.append({
                "title": title.strip() if title.strip() else "Untitled",
                "question": question.strip(),
                "answer": answer.strip()
            })

    return qa_list


if selected_mode == 'chat':
    st.write("You selected chat mode.")
    with st.form("chat_form"):
        question = st.text_input("Enter your message:")
        session_id_t = st.text_input("Enter session ID:")
        submit_button = st.form_submit_button("Submit")
        if submit_button:
            if question:
                if session_id_t:
                    session_history_t=get_session_history(session_id_t)
                    response_t = conversational_rag_chain_t.invoke(
                        {"input": question},
                        config={
                            "configurable": {"session_id":session_id_t}
                        },
                    )
                    st.write(response_t['answer'])
                else:
                    st.warning("Please enter a session ID.")
            else :
                st.warning("Please enter a question.")

elif selected_mode == 'split':
    st.write("You selected split mode.")
    with st.form("split-form"):
        prompt = st.text_input("Enter your message:")
        session_id = st.text_input("Enter session ID:")
        submit_button = st.form_submit_button("Submit")
        if submit_button:
            if prompt:
                if session_id:
                    session_history=get_session_history(session_id)
                    response = conversational_rag_chain.invoke(
                        {"input": prompt},
                        config={
                            "configurable": {"session_id":session_id}
                        },
                    )

                    ## GETTING THE TRANSCRIPT
                    data = {}
                    with open('./transcripts/podcast.json', 'r') as f:
                        data = json.load(f)
                    segments = data['segments']
                    words = [word for segment in segments for word in segment['words']]

                    text = response['answer']
                    qa_list = process_text(text)

                    # st.write(text)

                    for idx, qa in enumerate(qa_list, 1):
                        st.write(f"Q{idx}: {qa['question']}")
                        st.write(f"A{idx}: {qa['answer']}\n")
                   
                    delete_all_files_in_folder('videos/cut')

                    for idx, qa in enumerate(qa_list):                     
                        question = qa['question']
                        answer = qa['answer']

                        start, end, score = find_best_answer_time_range(answer, words)
                        clip_video('videos/Vision Transformer.mp4',start,end,f"{question}.mp4")
                    
                    show_clipped_videos()


