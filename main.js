// Initialize Lucide icons
lucide.createIcons();

// State Management
const state = {
    activePage: 'dashboard',
    currentCategory: '',
    currentQuestion: '',
    history: []
};

// Questions Pool
const questions = {
    'Behavioral': [
        "Tell me about a time you faced a significant challenge at work. How did you handle it?",
        "Describe a situation where you had to work with a difficult colleague.",
        "Give an example of a time you missed a deadline. What happened?"
    ],
    'Situational': [
        "If you were assigned a project with an unclear scope and tight deadline, what would you do?",
        "How would you handle receiving critical feedback from your manager?",
        "If two team members had a conflict affecting productivity, how would you step in?"
    ],
    'Leadership': [
        "Tell me about a time you led a project or initiative. What was the outcome?",
        "Describe a situation where you had to adapt to a major change at work.",
        "How do you motivate a team during a stressful period?"
    ]
};

// ─── Navigation ─────────────────────────────────────────────────────────────
document.querySelectorAll('.nav-links li').forEach(item => {
    item.addEventListener('click', () => {
        const pageId = item.getAttribute('data-page');
        goToPage(pageId);
    });
});

function goToPage(pageId) {
    // Update Nav
    document.querySelectorAll('.nav-links li').forEach(li => li.classList.remove('active'));
    document.querySelector(`[data-page="${pageId}"]`).classList.add('active');

    // Update Pages
    document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
    document.getElementById(pageId).classList.add('active');
    
    state.activePage = pageId;
}

// ─── Interview Logic ─────────────────────────────────────────────────────────
function startInterview(category) {
    state.currentCategory = category;
    document.getElementById('category-label').innerText = category;
    
    // Pick random question
    const pool = questions[category];
    state.currentQuestion = pool[Math.floor(Math.random() * pool.length)];
    document.getElementById('question-text').innerText = state.currentQuestion;
    
    // Reset inputs
    document.getElementById('answer-input').value = '';
    document.getElementById('feedback-container').classList.add('hidden');
    
    goToPage('interview');
}

async function evaluateResponse() {
    const answer = document.getElementById('answer-input').value;
    if (!answer.trim()) return alert("Please type your response first!");

    toggleLoading(true);

    try {
        const response = await fetch('http://localhost:5000/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: state.currentQuestion,
                answer: answer
            })
        });

        const data = await response.json();
        if (data.error) throw new Error(data.error);

        displayFeedback(data.result);
    } catch (err) {
        alert("Evaluation failed: " + err.message);
    } finally {
        toggleLoading(false);
    }
}

function displayFeedback(markdown) {
    const container = document.getElementById('feedback-container');
    const content = document.getElementById('feedback-content');
    const scorePill = document.getElementById('score-pill');

    // Simple markdown formatting for the demo
    let html = markdown
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');
    
    content.innerHTML = html;

    // Extract score if present
    const scoreMatch = markdown.match(/Score:\s*(\d)/i);
    if (scoreMatch) {
        scorePill.innerText = `Score: ${scoreMatch[1]}/5`;
        scorePill.className = `pill score-${scoreMatch[1]}`;
    }

    container.classList.remove('hidden');
    container.scrollIntoView({ behavior: 'smooth' });
}

function resetInterview() {
    startInterview(state.currentCategory);
}

function nextQuestion() {
    resetInterview();
}

// ─── Knowledge Base ──────────────────────────────────────────────────────────
async function askKnowledge() {
    const query = document.getElementById('knowledge-query').value;
    if (!query.trim()) return;

    toggleLoading(true);
    const resultBox = document.getElementById('knowledge-result');

    try {
        const response = await fetch('http://localhost:5000/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });

        const data = await response.json();
        resultBox.innerHTML = data.result.replace(/\n/g, '<br>');
        resultBox.classList.remove('hidden');
    } catch (err) {
        alert("Query failed: " + err.message);
    } finally {
        toggleLoading(false);
    }
}

function quickAsk(text) {
    document.getElementById('knowledge-query').value = text;
    askKnowledge();
}

// ─── UI Helpers ─────────────────────────────────────────────────────────────
function toggleLoading(show) {
    document.getElementById('loading-overlay').classList.toggle('hidden', !show);
}
