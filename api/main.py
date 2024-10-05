import logging
from os import environ
import pathlib

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from faster_whisper import WhisperModel
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch


load_dotenv("../.env")
app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(model_name="saved_models/USER-bge-m3", device=device)
audio_model = WhisperModel(
    "saved_models/faster-whisper-medium",
    compute_type="int8",
    device=device,
)
llm = ChatOpenAI(
    model=environ.get("OPENAI_MODEL"),
    openai_api_key=environ.get("OPENAI_API_KEY"),
    openai_api_base=environ.get("OPENAI_API_BASE_URL"),
    temperature=environ.get("OPENAI_TEMPERATURE"),
)
vectorstore = None
retriever = None


@app.post("/index_media")
async def index_media(media: UploadFile):
    pathlib.Path("data").mkdir(parents=True, exist_ok=True)
    with open(f"data/{media.filename}", "wb") as local_file:
        local_file.write(media.file.read())

    segments_n, _ = audio_model.transcribe(
        audio=f"data/{media.filename}",
        word_timestamps=False,
        condition_on_previous_text=False,
        vad_filter=False,
        beam_size=2,
        best_of=2,
        no_speech_threshold=0.7,
        without_timestamps=True,
    )

    documents = [
        Document(page_content=" ".join(segment.text for segment in segments_n))
    ]
    logging.warning(documents[0].page_content)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    return "Retriever is ready. Use /answer"


@app.post("/answer")
async def answer(question: str):
    if vectorstore is None or retriever is None:
        raise HTTPException(400, detail="Retriever is not ready. Use /index_media")

    retrieved_docs = retriever.invoke(question)
    logging.warning(retrieved_docs)
    messages = [
        SystemMessage(
            content=f"""Действуйте как помощник для ответов на вопросы.
Используйте следующие фрагменты транскрибации аудио, чтобы ответить на вопрос пользователя.
Если вы не знаете ответа, просто скажите, что не знаете ответ.
Используйте максимум три предложения и сделайте ответ максимально полезным и кратким.

Фрагменты трансрибации из аудио:
{retrieved_docs}
""",
        ),
        HumanMessage(content=question),
    ]
    answer = llm.invoke(messages)
    if isinstance(answer, AIMessage):
        answer = answer.content
    return answer.replace('"""', "").strip()
