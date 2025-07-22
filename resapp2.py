# ---------------LIBRARIES--------------
import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import nltk
import uuid
from qdrant_client import QdrantClient, models
from openai import AzureOpenAI
from typing import List
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
import stanza
import sys
import torch
sys.modules["torch.classes"] = None

# ---------------- SETUP ----------------
st.set_page_config(page_title="JD–Resume Matcher", layout="wide")
#nltk.download("punkt")
#nlp=stanza.download("en")
#nlp = stanza.Pipeline("en")

@st.cache_resource
def load_stanza_pipeline():
    return stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,constituency,depparse,sentiment,ner')

nlp = load_stanza_pipeline()

#Embedding Model to save in Qdrant
MODEL_OPTIONS = {
    "all-mpnet-base‑v2 (general‑purpose, 768‑d)": "all-mpnet-base-v2",
    "stsb-mpnet-base‑v2 (STS‑tuned, 768‑d)": "stsb-mpnet-base-v2"
}
selected_model_label = st.sidebar.selectbox("Select Embedding Model", list(MODEL_OPTIONS.keys()))
model_name = MODEL_OPTIONS[selected_model_label]
model = SentenceTransformer(model_name, device="cpu")
VECTOR_DIM = model.get_sentence_embedding_dimension()

# ---------------- COLLECTION SETUP ----------------
COLLECTION_RESUME = "resume_chunks"
COLLECTION_JD = "jd_chunks"

qdrant = QdrantClient(host="localhost", port=6333)

def ensure_collection_exists(collection_name):
    if collection_name not in [col.name for col in qdrant.get_collections().collections]:
        qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=VECTOR_DIM, distance=models.Distance.COSINE),
        )

ensure_collection_exists(COLLECTION_RESUME)
ensure_collection_exists(COLLECTION_JD)

# ---------------- GPT SETUP ----------------
client = AzureOpenAI(
    api_key="",
    api_version="2024-07-01-preview",
    azure_endpoint="https://azureopenaitest03.openai.azure.com/",
)
DEPLOYMENT_NAME = "gpt-4"

# ---------------- CHUNKING ----------------
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
semantic_chunker = SemanticChunker(
    embeddings=embeddings_model,
    sentence_split_regex=r"@@@SENTENCE_BOUNDARY@@@",
    buffer_size=0,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=0.8,
    min_chunk_size=200,
    add_start_index=True
)

@st.cache_data
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text= " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def semantic_chunk(text: str) -> List[str]:
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sentences if s.text.strip()]
    joined = " @@@SENTENCE_BOUNDARY@@@ ".join(sentences)
    chunks = semantic_chunker.create_documents([joined])
    return [chunk.page_content for chunk in chunks]

def process_and_upload(files, file_type):
    collection_name = COLLECTION_RESUME if file_type == "resume" else COLLECTION_JD
    for file in files:
        text = extract_text_from_pdf(file)
        chunks = list(dict.fromkeys(semantic_chunk(text)))

        embeddings = model.encode(chunks)
        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={
                    "chunk": chunk,
                    "file_name": file.name,
                    "type": file_type,
                    "chunk_index": idx
                },
            ))
        qdrant.upsert(collection_name=collection_name, points=points, wait=True)
        st.success(f"✅ Uploaded {len(points)} chunks from `{file.name}` into `{collection_name}`")

@st.cache_resource
def load_stored_resume_chunks():
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_RESUME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="type", match=models.MatchValue(value="resume"))]
        ),
        limit=10000
    )
    return points

@st.cache_resource
def load_stored_jd_chunks():
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_JD,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="type", match=models.MatchValue(value="jd"))]
        ),
        limit=10000
    )
    return points

# ---------------- UI ----------------
st.header("Step 0: View Stored Resumes in Qdrant")
stored_resume_chunks = load_stored_resume_chunks()
stored_filenames = sorted(set(pt.payload["file_name"] for pt in stored_resume_chunks))

stored_jd_chunks = load_stored_jd_chunks()
stored_jd_filenames = sorted(set(pt.payload["file_name"] for pt in stored_jd_chunks))

st.markdown("### 🗑️ Delete Resume Collection")
if st.checkbox("Yes, delete resumes?"):
    if st.button("Delete Resume Collection"):
        qdrant.delete_collection(collection_name=COLLECTION_RESUME)
        st.success("Deleted resume collection.")

if stored_filenames:
    selected_file = st.selectbox("📂 Select a stored resume to view chunks", stored_filenames)
    if selected_file:
        file_chunks = [pt.payload["chunk"] for pt in sorted(stored_resume_chunks,key=lambda pt: pt.payload["chunk_index"])if pt.payload["file_name"] == selected_file]
        st.write(f"🔹 Found `{len(file_chunks)}` chunks in `{selected_file}`")
        with st.expander("📄 View Resume Chunks"):
            for idx, chunk in enumerate(file_chunks):
                st.markdown(f"**Chunk {idx + 1}:** {chunk}")
else:
    st.info("No resume data found in Qdrant yet.")

if stored_jd_filenames:
    selected_jd_file = st.selectbox("📂 Select a stored jd to view chunks", stored_jd_filenames)
    if selected_jd_file:
        file_chunks = [pt.payload["chunk"] for pt in sorted(stored_jd_chunks,key=lambda pt: pt.payload["chunk_index"])if pt.payload["file_name"] == selected_jd_file]
        st.write(f"🔹 Found `{len(file_chunks)}` chunks in `{selected_jd_file}`")
        with st.expander("📄 View JD Chunks"):
            for idx, chunk in enumerate(file_chunks):
                st.markdown(f"**Chunk {idx + 1}:** {chunk}")
else:
    st.info("No jd found in Qdrant yet.")

# ---------------- Upload PDFs MAIN----------------
st.header("Step 1: Upload PDFs")
col1, col2 = st.columns(2)
with col1:
    resumes = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)
with col2:
    jds = st.file_uploader("Upload Job Descriptions (PDF)", type="pdf", accept_multiple_files=True)

if resumes:
    process_and_upload(resumes, "resume")
if jds:
    process_and_upload(jds, "jd")

# ---------------- Query ----------------
st.header("Step 2: Custom Query with JD + GPT-4 Answer")

if stored_jd_filenames:
    selected_jd_file = st.selectbox("📎 Choose a JD from stored files", stored_jd_filenames)
else:
    st.warning("⚠️ No job descriptions found in the database. Please upload one.")
    selected_jd_file = None

with st.form("query_form"):
    query_input = st.text_input("🔍 Enter your query (e.g., 'Find candidate for NLP and Azure')")
    submitted = st.form_submit_button("Submit")


if submitted and query_input and selected_jd_file:
    jd_chunks_for_selected_file = [
        pt.payload["chunk"] for pt in qdrant.scroll(
            collection_name=COLLECTION_JD,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="type", match=models.MatchValue(value="jd")),
                    models.FieldCondition(key="file_name", match=models.MatchValue(value=selected_jd_file))
                ]
            ),
            limit=100
        )[0]
    ]
    selected_jd_text = "\n".join(jd_chunks_for_selected_file)

    jd_summary_prompt = f"""
Summarize the following job description into 3-5 bullet points that highlight the key responsibilities and required skills:

{selected_jd_text}
"""
    with st.spinner("🧠 Summarizing JD using GPT-4..."):
        jd_summary = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an expert HR assistant."},
                {"role": "user", "content": jd_summary_prompt}
            ],
            temperature=0.3,
            max_tokens=300
        ).choices[0].message.content

    st.markdown("### 📄 JD Summary")
    st.markdown(jd_summary)

    combined_query_text = f"Role: Recruiter. Query: {query_input}\n\nRelevant JD Chunks:\n" + "\n".join(jd_chunks_for_selected_file)
    query_embedding = model.encode([combined_query_text])[0].tolist()

    results = qdrant.search(
        collection_name=COLLECTION_RESUME,
        query_vector=query_embedding,
        limit=1,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="type", match=models.MatchValue(value="resume"))]
        ),
    )

    top_file = results[0].payload["file_name"]
    top_resume_chunks = [pt.payload["chunk"] for pt in stored_resume_chunks if pt.payload["file_name"] == top_file]
    full_resume_text = "\n".join(top_resume_chunks)

    st.markdown(f"### 🔍 Best Match: `{top_file}`")

    prompt = f"""
You are a technical recruiter.

## Job Description Summary:
{jd_summary}

## Resume (Full Text):
{full_resume_text}

## Hiring Manager Query:
"{query_input}"

Instructions:
- Only use resume facts. No assumptions.
- Highlight supporting evidence.
- Conclude with: Good Fit / Partial Fit / Not a Fit.
- Analyze how well the candidate’s resume aligns with the provided job description and give a fit score out of 100. Use the following breakdown:
Core Skills & Expertise (/20), Tools & Methodologies (/20), Domain Knowledge & Relevance (/20), Innovation & Problem Solving (/20), and Communication & Impact (/20).
Provide brief feedback summarizing key strengths and any gaps.
"""
    with st.spinner("🔍 Evaluating resume using GPT-4..."):
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a detail-oriented AI recruiter."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=800
        )
        st.markdown("### 💡 GPT-4 Evaluation")
        st.markdown(response.choices[0].message.content)
