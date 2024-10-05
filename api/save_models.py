from faster_whisper.utils import download_model
from sentence_transformers import SentenceTransformer

st_model = SentenceTransformer("deepvk/USER-bge-m3")
st_model.save("saved_models/USER-bge-m3")

download_model("medium", output_dir="saved_models/faster-whisper-medium")
