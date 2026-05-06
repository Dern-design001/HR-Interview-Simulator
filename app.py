"""
app.py — HR Interview Simulator
Streamlit frontend powered by Google Gemini + ChromaDB RAG
"""

import os
import time
import random
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ─── Page config (MUST be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="HR Interview Simulator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&display=swap');

/* ── Reset & base ──────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App background ──────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1040 40%, #0d1b3e 100%);
    min-height: 100vh;
}

/* ── Sidebar ──────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* ── Main title ──────────────────────────────────────────── */
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

/* ── Glass cards ─────────────────────────────────────────── */
.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1.2rem;
    transition: border 0.3s;
}
.glass-card:hover {
    border-color: rgba(167,139,250,0.4);
}

/* ── Question card ───────────────────────────────────────── */
.question-card {
    background: linear-gradient(135deg, rgba(167,139,250,0.12), rgba(96,165,250,0.08));
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.question-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}
.question-text {
    font-size: 1.15rem;
    font-weight: 500;
    color: #f1f5f9;
    line-height: 1.55;
}

/* ── Feedback card ───────────────────────────────────────── */
.feedback-card {
    background: rgba(52,211,153,0.07);
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
}
.feedback-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #34d399;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.6rem;
}
.feedback-text {
    color: #cbd5e1;
    line-height: 1.7;
    font-size: 0.96rem;
}

/* ── Score badge ─────────────────────────────────────────── */
.score-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.9rem;
    margin-right: 0.5rem;
}
.score-5 { background: rgba(52,211,153,0.2); color: #34d399; border: 1px solid #34d399; }
.score-4 { background: rgba(96,165,250,0.2); color: #60a5fa; border: 1px solid #60a5fa; }
.score-3 { background: rgba(251,191,36,0.2); color: #fbbf24; border: 1px solid #fbbf24; }
.score-2 { background: rgba(251,146,60,0.2); color: #fb923c; border: 1px solid #fb923c; }
.score-1 { background: rgba(248,113,113,0.2); color: #f87171; border: 1px solid #f87171; }

/* ── Progress bar override ───────────────────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #a78bfa, #60a5fa) !important;
    border-radius: 999px;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.55rem 1.6rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.5) !important;
}

/* ── Text area ───────────────────────────────────────────── */
.stTextArea textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    font-size: 0.96rem !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus {
    border-color: rgba(167,139,250,0.5) !important;
    box-shadow: 0 0 0 2px rgba(167,139,250,0.15) !important;
}

/* ── Select box ──────────────────────────────────────────── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}

/* ── Metric ──────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 0.8rem 1rem;
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
[data-testid="stMetricValue"] { color: #a78bfa !important; font-family: 'Outfit', sans-serif; }

/* ── Divider ─────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Spinner / status text ───────────────────────────────── */
.stSpinner > div { border-top-color: #a78bfa !important; }

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.3); border-radius: 999px; }

/* ── Chip ────────────────────────────────────────────────── */
.chip {
    display: inline-block;
    background: rgba(167,139,250,0.15);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 999px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    color: #a78bfa;
    font-weight: 500;
    margin: 0.2rem;
}

/* ── History entry ───────────────────────────────────────── */
.history-q { color: #94a3b8; font-size: 0.85rem; margin-bottom: 0.3rem; font-style: italic; }
.history-a { color: #cbd5e1; font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────
CHROMA_PATH = "./chromadb_storage"
COLLECTION  = "hr_rubric"

CATEGORIES = [
    "All Topics",
    "Opening / Introduction",
    "Behavioral Questions",
    "Situational Questions",
    "Closing Questions",
    "DEI & Legal Compliance",
    "Evaluation & Rubric",
]

SAMPLE_QUESTIONS = {
    "Opening / Introduction": [
        "Tell me about yourself and your professional background.",
        "Why are you interested in this role and our company?",
    ],
    "Behavioral Questions": [
        "Tell me about a time you faced a significant challenge at work. How did you handle it?",
        "Describe a situation where you had to work with a difficult colleague. What did you do?",
        "Give an example of a time you missed a deadline. What happened and what did you learn?",
        "Tell me about a time you led a project or initiative. What was the outcome?",
        "Describe a situation where you had to adapt to a major change at work.",
    ],
    "Situational Questions": [
        "If you were assigned a project with an unclear scope and tight deadline, what would you do?",
        "How would you handle receiving critical feedback from your manager?",
        "If two team members had a conflict affecting productivity, how would you step in?",
    ],
    "Closing Questions": [
        "Where do you see yourself in 3–5 years?",
        "What questions do you have for us about the role or company?",
    ],
    "DEI & Legal Compliance": [
        "What DEI considerations should an interviewer keep in mind?",
        "Which questions are legally prohibited in a job interview and why?",
        "How can structured interviews reduce unconscious bias?",
    ],
    "Evaluation & Rubric": [
        "What does a score of 5 mean on the HR evaluation rubric?",
        "What is the overall hiring recommendation threshold for a 'Strong Hire'?",
        "How should an interviewer document feedback after an interview?",
    ],
}

ALL_QUESTIONS = [q for qs in SAMPLE_QUESTIONS.values() for q in qs]

# ─── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_chain():
    """Load the RAG chain once and cache it."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None, "❌ GOOGLE_API_KEY not found. Add it to your `.env` file."

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            collection_name=COLLECTION,
            embedding_function=embeddings,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.4,
        )

        prompt_template = """You are an expert HR interview coach and evaluator.
Use the following HR rubric context to evaluate the candidate's answer or answer the HR-related question.

Context from HR Rubric:
{context}

Question / Candidate Answer to Evaluate:
{question}

Provide:
1. A clear, constructive evaluation (if evaluating an answer) OR a direct, expert answer (if asked an HR question).
2. Specific strengths and areas for improvement (when evaluating).
3. A score from 1–5 with justification (when evaluating a candidate's interview answer).
4. A brief, actionable tip to improve.

Keep your response professional, encouraging, and structured.
"""
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=False,
        )
        return chain, None
    except Exception as e:
        return None, f"❌ Failed to load AI chain: {e}"


def get_score_class(score: int) -> str:
    return f"score-{min(max(score, 1), 5)}"


def extract_score(text: str) -> int | None:
    """Try to pull a numeric score 1–5 from AI feedback text."""
    import re
    patterns = [
        r"score[:\s]+([1-5])",
        r"([1-5])\s*/\s*5",
        r"rating[:\s]+([1-5])",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


# ─── Session state defaults ───────────────────────────────────────────────────
def init_state():
    defaults = {
        "history": [],          # list of {question, answer, feedback, score}
        "current_q": "",
        "mode": "Practice",
        "total_score": 0,
        "question_count": 0,
        "mock_mode_active": False,
        "mock_q_index": 0,
        "mock_questions": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 HR Simulator")
    st.markdown("---")

    mode = st.radio(
        "**Mode**",
        ["Practice", "Mock Interview", "HR Knowledge Q&A"],
        index=["Practice", "Mock Interview", "HR Knowledge Q&A"].index(st.session_state.mode),
    )
    st.session_state.mode = mode

    st.markdown("---")

    if mode == "Practice":
        category = st.selectbox("**Category**", CATEGORIES)
        if category == "All Topics":
            pool = ALL_QUESTIONS
        else:
            pool = SAMPLE_QUESTIONS.get(category, ALL_QUESTIONS)

        if st.button("🎲 Random Question"):
            st.session_state.current_q = random.choice(pool)

    elif mode == "Mock Interview":
        num_q = st.slider("Number of questions", 3, 12, 6)
        if st.button("▶️ Start Mock Interview"):
            st.session_state.mock_questions = random.sample(
                ALL_QUESTIONS, min(num_q, len(ALL_QUESTIONS))
            )
            st.session_state.mock_q_index = 0
            st.session_state.mock_mode_active = True
            st.session_state.history = []
            st.session_state.total_score = 0
            st.session_state.question_count = 0
            st.session_state.current_q = st.session_state.mock_questions[0]

    st.markdown("---")

    # Stats
    if st.session_state.question_count > 0:
        avg = st.session_state.total_score / st.session_state.question_count
        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        col1.metric("Questions", st.session_state.question_count)
        col2.metric("Avg Score", f"{avg:.1f}/5")
        st.progress(avg / 5)

    st.markdown("---")
    if st.button("🔄 Reset Session"):
        for key in ["history", "current_q", "total_score", "question_count",
                    "mock_mode_active", "mock_q_index", "mock_questions"]:
            st.session_state[key] = [] if key in ["history", "mock_questions"] else (
                0 if key in ["total_score", "question_count", "mock_q_index"] else
                False if key == "mock_mode_active" else ""
            )
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='color:#475569;font-size:0.75rem;'>Powered by Google Gemini + ChromaDB<br>© 2025 HR Interview Simulator</div>",
        unsafe_allow_html=True,
    )

# ─── Main content ─────────────────────────────────────────────────────────────
# Header
st.markdown('<p class="hero-title">🎯 HR Interview Simulator</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Practice interviews, get AI-powered feedback, and master HR questions — all grounded in your rubric.</p>',
    unsafe_allow_html=True,
)

# Load chain
with st.spinner("Loading AI engine …"):
    chain, error = load_chain()

if error:
    st.error(error)
    st.info("**Setup steps:**\n1. Add your `GOOGLE_API_KEY` to `.env`\n2. Run `python ingest.py`\n3. Restart the app")
    st.stop()

# ── Mode: HR Knowledge Q&A ────────────────────────────────────────────────────
if st.session_state.mode == "HR Knowledge Q&A":
    st.markdown("### 🧠 Ask Any HR / Interview Question")
    st.markdown('<div class="glass-card">Ask anything about interview best practices, the rubric, DEI guidelines, legal compliance, scoring, etc.</div>', unsafe_allow_html=True)

    faq_cols = st.columns(3)
    faq_examples = [
        "What does the STAR method mean?",
        "Which interview questions are illegal?",
        "How to reduce unconscious bias?",
        "What score is a 'Strong Hire'?",
        "How should I document interview feedback?",
        "What are DEI best practices?",
    ]
    for i, faq in enumerate(faq_examples):
        with faq_cols[i % 3]:
            if st.button(faq, key=f"faq_{i}"):
                st.session_state.current_q = faq

    hr_question = st.text_area(
        "Your question:",
        value=st.session_state.current_q,
        placeholder="e.g. How should I evaluate a candidate's problem-solving skills?",
        height=100,
        key="hr_q_input",
    )

    if st.button("🔍 Get Answer", key="hr_ask"):
        if hr_question.strip():
            with st.spinner("Consulting HR knowledge base …"):
                result = chain.invoke({"query": hr_question})
                answer = result.get("result", "No answer returned.")
            st.markdown('<div class="feedback-card"><div class="feedback-label">💡 Expert Answer</div><div class="feedback-text">' + answer.replace("\n", "<br>") + '</div></div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a question.")

# ── Mode: Practice & Mock Interview ──────────────────────────────────────────
else:
    # Mock interview progress bar
    if st.session_state.mock_mode_active and st.session_state.mock_questions:
        total = len(st.session_state.mock_questions)
        done  = st.session_state.mock_q_index
        st.markdown(f"**Mock Interview Progress** — Question {done + 1} of {total}")
        st.progress((done) / total)
        st.markdown("---")

    # Question display
    if st.session_state.current_q:
        st.markdown(
            f'<div class="question-card"><div class="question-label">📋 Interview Question</div>'
            f'<div class="question-text">{st.session_state.current_q}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:2.5rem;">'
            '<span style="font-size:2.5rem;">💬</span><br>'
            '<span style="color:#94a3b8;font-size:1rem;">Select a category and click <b>Random Question</b> in the sidebar, or start a Mock Interview.</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Answer input
    answer = st.text_area(
        "✍️ Your Answer:",
        placeholder="Type your answer here. Use the STAR method: Situation → Task → Action → Result.",
        height=180,
        key="answer_input",
    )

    col_submit, col_skip = st.columns([2, 1])

    with col_submit:
        submit = st.button("📤 Submit Answer for Feedback", disabled=not (st.session_state.current_q and answer.strip()))

    with col_skip:
        if st.session_state.mock_mode_active:
            skip = st.button("⏭️ Skip Question")
        else:
            skip = False

    # Process submission
    if submit and st.session_state.current_q and answer.strip():
        query = (
            f"Interview Question: {st.session_state.current_q}\n\n"
            f"Candidate's Answer: {answer.strip()}\n\n"
            "Please evaluate this answer against the HR rubric and provide structured feedback with a score from 1–5."
        )
        with st.spinner("🤖 Analyzing your answer …"):
            result = chain.invoke({"query": query})
            feedback = result.get("result", "Unable to generate feedback.")

        score = extract_score(feedback)

        # Store in history
        st.session_state.history.append({
            "question": st.session_state.current_q,
            "answer": answer.strip(),
            "feedback": feedback,
            "score": score,
        })

        if score:
            st.session_state.total_score += score
        st.session_state.question_count += 1

        # Render feedback
        score_html = ""
        if score:
            sc = get_score_class(score)
            score_html = f'<span class="score-badge {sc}">Score: {score}/5</span>'

        st.markdown(
            f'<div class="feedback-card">'
            f'<div class="feedback-label">🤖 AI Feedback {score_html}</div>'
            f'<div class="feedback-text">{feedback.replace(chr(10), "<br>")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Advance mock interview
        if st.session_state.mock_mode_active:
            next_idx = st.session_state.mock_q_index + 1
            if next_idx < len(st.session_state.mock_questions):
                st.session_state.mock_q_index = next_idx
                st.session_state.current_q = st.session_state.mock_questions[next_idx]
                time.sleep(0.5)
                st.rerun()
            else:
                st.session_state.mock_mode_active = False
                st.balloons()
                st.success("🎉 Mock interview complete! Review your history below.")

    # Skip logic
    if skip:
        next_idx = st.session_state.mock_q_index + 1
        if next_idx < len(st.session_state.mock_questions):
            st.session_state.mock_q_index = next_idx
            st.session_state.current_q = st.session_state.mock_questions[next_idx]
            st.rerun()
        else:
            st.session_state.mock_mode_active = False
            st.info("Mock interview ended (all questions skipped or answered).")

# ─── Answer History ───────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📚 Session History")

    for i, entry in enumerate(reversed(st.session_state.history)):
        idx = len(st.session_state.history) - i
        score_badge = ""
        if entry["score"]:
            sc = get_score_class(entry["score"])
            score_badge = f'<span class="score-badge {sc}">{entry["score"]}/5</span>'

        with st.expander(f"Q{idx}: {entry['question'][:70]}… {entry['score'] and f'[{entry[\"score\"]}/5]' or ''}"):
            st.markdown(f'<p class="history-q">📋 {entry["question"]}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="history-a"><b>Your answer:</b> {entry["answer"]}</p>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="feedback-card" style="margin-top:0.8rem;">'
                f'<div class="feedback-label">Feedback {score_badge}</div>'
                f'<div class="feedback-text">{entry["feedback"].replace(chr(10), "<br>")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
