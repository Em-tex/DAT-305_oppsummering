/* =========================================
   1. NAVIGATION & UI LOGIC
   ========================================= */

function showSection(id, element) {
    // Hide all modules and quiz
    document.querySelectorAll('.module').forEach(mod => mod.classList.remove('active'));
    document.getElementById('quiz-area').classList.remove('active');
    document.getElementById('content-area').style.display = 'block';
    
    // Update Navbar
    document.querySelectorAll('.nav-links li').forEach(li => li.classList.remove('active'));
    element.classList.add('active');

    // Show selected module and scroll top
    document.getElementById(id).classList.add('active');
    window.scrollTo(0, 0);
}

function showQuiz(element) {
    // Hide modules, Show Quiz
    document.getElementById('content-area').style.display = 'none';
    document.getElementById('quiz-area').classList.add('active');
    
    // Update Navbar
    document.querySelectorAll('.nav-links li').forEach(li => li.classList.remove('active'));
    element.classList.add('active');
    
    // Start quiz if not started
    if(quizState.currentIdx === 0 && quizState.score === 0) {
        initQuiz();
    }
}

// Accordion Logic
function toggleAcc(element) {
    const content = element.nextElementSibling;
    content.classList.toggle('open');
    const icon = element.querySelector('i');
    if (content.classList.contains('open')) {
        icon.classList.remove('fa-chevron-down');
        icon.classList.add('fa-chevron-up');
    } else {
        icon.classList.remove('fa-chevron-up');
        icon.classList.add('fa-chevron-down');
    }
}

// Pop-up / Modal Logic
document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById("term-modal");
    const closeBtn = document.querySelector(".close");
    
    document.querySelectorAll('.term').forEach(term => {
        term.addEventListener('click', () => {
            document.getElementById('modal-title').innerText = term.innerText;
            document.getElementById('modal-desc').innerText = term.getAttribute('data-desc');
            modal.style.display = "block";
        });
    });

    closeBtn.onclick = () => modal.style.display = "none";
    window.onclick = (event) => {
        if (event.target == modal) modal.style.display = "none";
    }
});


/* =========================================
   2. QUIZ ENGINE
   ========================================= */

let quizState = {
    questions: [],
    currentIdx: 0,
    score: 0
};

function initQuiz() {
    // Access the global variable from questions.js
    const db = window.rawQuestions;

    if (!db || db.length === 0) {
        document.getElementById('quiz-card-container').innerHTML = "<p style='color:red;'>Error: Questions not loaded. Check questions.js.</p>";
        return;
    }

    // Clone and Shuffle
    let qList = [...db].sort(() => 0.5 - Math.random());
    
    // Select top 60 (or all if less)
    quizState.questions = qList.slice(0, 60); 
    quizState.currentIdx = 0;
    quizState.score = 0;

    renderQuestion();
}

function renderQuestion() {
    if(quizState.currentIdx >= quizState.questions.length) {
        showSummary();
        return;
    }

    const q = quizState.questions[quizState.currentIdx];
    
    // Stats
    document.getElementById('q-current').innerText = quizState.currentIdx + 1;
    document.getElementById('q-total').innerText = quizState.questions.length;
    document.getElementById('score').innerText = quizState.score;

    // Buttons
    document.getElementById('submit-btn').classList.remove('hidden');
    document.getElementById('next-btn').classList.add('hidden');

    // Build HTML
    let html = `
        <div class="question-card">
            <h3>${q.q}</h3>
            <div class="options-list">
    `;
    
    q.options.forEach((opt, idx) => {
        html += `<div class="option-btn" onclick="selectOption(this, ${idx})">${opt}</div>`;
    });

    html += `
            </div>
            <div class="explanation hidden" id="explanation">
                <strong>Explanation:</strong> ${q.exp}
            </div>
        </div>
    `;

    document.getElementById('quiz-card-container').innerHTML = html;
}

function selectOption(btn, idx) {
    if(!document.getElementById('next-btn').classList.contains('hidden')) return; // Locked

    document.querySelectorAll('.option-btn').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    btn.dataset.idx = idx;
}

function checkAnswer() {
    const selected = document.querySelector('.option-btn.selected');
    if(!selected) { alert("Select an answer!"); return; }

    const q = quizState.questions[quizState.currentIdx];
    const userAns = parseInt(selected.dataset.idx);

    document.querySelectorAll('.option-btn').forEach((btn, idx) => {
        if(idx === q.a) btn.classList.add('correct');
        if(idx === userAns && idx !== q.a) btn.classList.add('wrong');
    });

    if(userAns === q.a) quizState.score++;

    document.getElementById('explanation').classList.remove('hidden');
    document.getElementById('submit-btn').classList.add('hidden');
    document.getElementById('next-btn').classList.remove('hidden');
    
    // Update score immediately
    document.getElementById('score').innerText = quizState.score;
}

function nextQuestion() {
    quizState.currentIdx++;
    renderQuestion();
}

function showSummary() {
    const pct = Math.round((quizState.score / quizState.questions.length) * 100);
    document.getElementById('quiz-card-container').innerHTML = `
        <div class="slide" style="text-align: center;">
            <h2>Quiz Finished!</h2>
            <div style="font-size: 4rem; color: var(--primary); margin: 20px 0; font-weight:bold;">${pct}%</div>
            <p>You scored ${quizState.score} out of ${quizState.questions.length}</p>
            <button class="btn-primary" onclick="initQuiz()">Restart Quiz</button>
        </div>
    `;
    document.getElementById('quiz-footer').classList.add('hidden');
}