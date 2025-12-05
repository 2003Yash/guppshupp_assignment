import streamlit as st
import time
import json
import uuid
from typing import List, Dict, Any, Tuple
from huggingface_hub import InferenceClient

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG
# ============================================================
import streamlit as st

MY_HF_TOKEN = st.secrets["HF_TOKEN"]
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
client = InferenceClient(MODEL_NAME, token=MY_HF_TOKEN)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

PERSONAS = {
    "Witty Friend": "funny, casual, uses humor and metaphors, warm tone",
    "Stoic Mentor": "calm, wise, motivational, philosophical",
    "Therapist": "empathetic, reflective, non-judgmental, supportive"
}

# FIXED PROMPTS (with escaped curly braces)
EXTRACTION_PROMPT = """
Analyze the following chat messages and extract key memory points about the user.
Return ONLY a JSON array of memory objects like this:
[
  {{"category": "preference", "text": "Loves playing football"}},
  {{"category": "fact", "text": "Lives in Bangalore"}},
  {{"category": "emotion", "text": "Stressed about work"}}
]

CHAT MESSAGES:
{chat}

JSON OUTPUT:
"""

VALIDATION_PROMPT = """
Review these memory candidates and score each one.
Return JSON array with:
- keep: true/false
- confidence: 0.0-1.0
- importance: 0.0-1.0

CANDIDATES:
{candidates}

JSON OUTPUT:
"""

CHAT_PROMPT = """
You are chatting as: {persona_name}
Persona style: {persona_desc}

Relevant user memories:
{memories}

User says: "{user_message}"

Respond in persona style in under 3 sentences.
Response:
"""

# ============================================================
# LLM CALL FUNCTION
# ============================================================
def call_llm(prompt, max_tokens=300, temperature=0.2):
    try:
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
            # PRINT THE ERROR ON SCREEN
            st.error(f"❌ CONNECTION ERROR: {e}")
            
            # Fallback: detailed print to terminal
            print(f"FAILED PROMPT: {prompt[:50]}...")
            print(f"ERROR DETAILS: {e}")
            return ""

# ============================================================
# MEMORY VECTOR STORE
# ============================================================
class VectorMemory:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids = []
        self.meta = {}

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        return v / (norm + 1e-10)

    def add(self, text, metadata):
        emb = embedder.encode(text, convert_to_numpy=True)
        emb = self._normalize(emb).astype("float32").reshape(1, -1)

        mem_id = str(uuid.uuid4())
        self.index.add(emb)
        self.ids.append(mem_id)
        self.meta[mem_id] = {"text": text, **metadata}

    def search(self, query, k=5):
        if self.index.ntotal == 0:
            return []

        q = embedder.encode(query, convert_to_numpy=True)
        q = self._normalize(q).astype("float32").reshape(1, -1)

        scores, idxs = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            mem_id = self.ids[idx]
            results.append((score, self.meta[mem_id]))
        return results

# ============================================================
# MEMORY EXTRACTION
# ============================================================
def extract_memories(chat_history):
    text = "\n".join(f"{m['role']}: {m['content']}" for m in chat_history)
    raw = call_llm(EXTRACTION_PROMPT.format(chat=text), max_tokens=600)

    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except:
        return []

def validate_memories(candidates):
    if not candidates:
        return []

    raw = call_llm(
        VALIDATION_PROMPT.format(candidates=json.dumps(candidates)),
        max_tokens=400
    )

    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        scores = json.loads(cleaned)
        out = []
        for cand, score in zip(candidates, scores):
            if score.get("keep", True):
                out.append({
                    "category": cand["category"],
                    "text": cand["text"],
                    "confidence": score.get("confidence", 0.7),
                    "importance": score.get("importance", 0.5),
                })
        return out
    except:
        return candidates

# ============================================================
# CHAT WITH MEMORY + PERSONA
# ============================================================
def chat_with_persona(message, persona, memory_store):
    relevant = memory_store.search(message, k=5)
    mem_lines = [
        f"- {m['text']} ({m['category']})"
        for score, m in relevant if score > 0.3
    ]

    mem_text = "\n".join(mem_lines) if mem_lines else "None."

    prompt = CHAT_PROMPT.format(
        persona_name=persona,
        persona_desc=PERSONAS[persona],
        memories=mem_text,
        user_message=message
    )

    return call_llm(prompt, temperature=0.5)

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Memory Persona Chatbot", layout="wide")

st.title("🧠 Memory-Aware Persona Chatbot")

with st.sidebar:
    st.header("⚙️ Persona Selection")
    persona = st.selectbox("Choose Persona", list(PERSONAS.keys()))

st.write("This bot extracts memories from past chats and responds using the selected persona.")

# ------------------------------------------------------------
# STEP 1: Extract Memories at App Start
# ------------------------------------------------------------
CHAT_HISTORY = [
    {"role": "user", "content": "I love playing football on weekends."},
    {"role": "user", "content": "I'm feeling stressed about work lately."},
    {"role": "user", "content": "My dog Bruno keeps chewing my shoes."},
    {"role": "user", "content": "I prefer quiet cafes."},
    {"role": "user", "content": "I hate crowded places."},
    {"role": "user", "content": "I'm excited about my new laptop."},
    {"role": "user", "content": "I live in Bangalore."},
    {"role": "user", "content": "I like watching sci-fi movies."},
    {"role": "user", "content": "I work as a software developer."},
    {"role": "user", "content": "I love cold weather."},
    {"role": "user", "content": "I prefer tea over coffee."},
    {"role": "user", "content": "I'm anxious about tomorrow's meeting."},
    {"role": "user", "content": "My sister is visiting next week."},
    {"role": "user", "content": "I like learning new languages."},
    {"role": "user", "content": "I have a small garden."},
    {"role": "user", "content": "I hate slow internet."},
    {"role": "user", "content": "I'm feeling motivated today."},
    {"role": "user", "content": "I love cooking Italian food."},
    {"role": "user", "content": "My car broke down yesterday."},
    {"role": "user", "content": "I feel tired recently."},
    {"role": "user", "content": "I prefer working at night."},
    {"role": "user", "content": "I live with two roommates."},
    {"role": "user", "content": "I like hiking."},
    {"role": "user", "content": "I love listening to jazz."},
    {"role": "user", "content": "I hate alarm clocks."},
    {"role": "user", "content": "I'm excited for my trip next month."},
    {"role": "user", "content": "I feel overwhelmed sometimes."},
    {"role": "user", "content": "My parents live nearby."},
    {"role": "user", "content": "I love minimalistic design."},
    {"role": "user", "content": "I prefer online shopping."},
]

if "memory_store" not in st.session_state:
    st.session_state.memory_store = VectorMemory(embedder.get_sentence_embedding_dimension())

    with st.spinner("Extracting memories from 30 chat messages..."):
        candidates = extract_memories(CHAT_HISTORY)
        validated = validate_memories(candidates)

        for mem in validated:
            st.session_state.memory_store.add(
                mem["text"],
                {
                    "category": mem.get("category", "unknown"),
                    "confidence": mem.get("confidence", 0.7),
                    "importance": mem.get("importance", 0.5),
                }
            )

    st.success(f"Stored {len(validated)} memories!")

# ------------------------------------------------------------
# RIGHT SIDE CHAT INTERFACE
# ------------------------------------------------------------
st.subheader("💬 Chat with Persona")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

user_input = st.chat_input("Say something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        reply = chat_with_persona(user_input, persona, st.session_state.memory_store)
        st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
