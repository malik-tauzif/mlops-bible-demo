from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np

app = FastAPI()

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

documents = [
    "Refund Policy: Students may request a refund within 14 days of enrollment.",
    "Course materials include video lectures, hands-on labs, and a final project.",
    "Office hours are held every Tuesday and Thursday from 6 PM to 8 PM IST.",
]
doc_embeddings = embed_model.encode(documents)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

BLOCKED_PATTERNS = ["ignore your instructions", "admin password"]

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "RAG chatbot API is live."}

@app.post("/chat")
def chat(request: ChatRequest):
    question = request.question

    if any(p in question.lower() for p in BLOCKED_PATTERNS):
        return {"answer": "I can't help with that request."}

    q_embedding = embed_model.encode([question])
    _, indices = index.search(np.array(q_embedding), k=1)
    context = documents[indices[0][0]]

    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=60)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"answer": answer, "retrieved_context": context}
