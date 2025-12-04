// --- NAVIGATION & UI LOGIC ---
function showSection(id, element) {
    document.querySelectorAll('.module').forEach(mod => mod.classList.remove('active'));
    document.getElementById('quiz-area').classList.remove('active');
    document.getElementById('content-area').style.display = 'block';
    
    document.querySelectorAll('.nav-links li').forEach(li => li.classList.remove('active'));
    element.classList.add('active');

    document.getElementById(id).classList.add('active');
    window.scrollTo(0, 0);
}

function showQuiz(element) {
    document.getElementById('content-area').style.display = 'none';
    document.getElementById('quiz-area').classList.add('active');
    
    document.querySelectorAll('.nav-links li').forEach(li => li.classList.remove('active'));
    element.classList.add('active');
    
    if(quizState.currentIdx === 0 && quizState.score === 0) {
        initQuiz();
    }
}

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
    window.onclick = (event) => { if(event.target == modal) modal.style.display = "none"; }
});

// --- QUIZ ENGINE LOGIC ---
let quizState = { questions: [], currentIdx: 0, score: 0 };

function initQuiz() {
    // Sjekk om spørsmål er lastet inn
    if (!window.rawQuestions) {
        alert("Feil: Spørsmål ble ikke lastet inn. Sjekk at questions.js ligger i mappen.");
        return;
    }

    // Bland spørsmålene og fyll på hvis det er for få (simulering)
    let expanded = [...window.rawQuestions];
    while(expanded.length < 60) {
        let r = window.rawQuestions[Math.floor(Math.random() * window.rawQuestions.length)];
        expanded.push({...r, q: "Review: " + r.q});
    }

    quizState.questions = expanded.sort(() => 0.5 - Math.random());
    quizState.currentIdx = 0;
    quizState.score = 0;
    renderQuestion();
}

function renderQuestion() {
    if(quizState.currentIdx >= quizState.questions.length) { showSummary(); return; }
    
    const q = quizState.questions[quizState.currentIdx];
    const container = document.getElementById('quiz-card-container');
    
    document.getElementById('q-current').innerText = quizState.currentIdx + 1;
    document.getElementById('q-total').innerText = quizState.questions.length;
    document.getElementById('score').innerText = quizState.score;
    document.getElementById('submit-btn').classList.remove('hidden');
    document.getElementById('next-btn').classList.add('hidden');

    let html = `<div class="question-card"><h3>${q.q}</h3><div class="options-list">`;
    q.options.forEach((opt, idx) => { html += `<div class="option-btn" onclick="selectOption(this, ${idx})">${opt}</div>`; });
    html += `</div><div class="explanation hidden" id="explanation"><strong>Explanation:</strong> ${q.exp}</div></div>`;
    container.innerHTML = html;
}

function selectOption(btn, idx) {
    if(!document.getElementById('next-btn').classList.contains('hidden')) return;
    document.querySelectorAll('.option-btn').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    btn.dataset.idx = idx;
}

function checkAnswer() {
    const selected = document.querySelector('.option-btn.selected');
    if(!selected) { alert("Please select an answer."); return; }
    
    const q = quizState.questions[quizState.currentIdx];
    const userAns = parseInt(selected.dataset.idx);

    document.querySelectorAll('.option-btn').forEach((btn, idx) => {
        if(idx === q.a) btn.classList.add('correct');
        if(idx === userAns && idx !== q.a) btn.classList.add('wrong');
    });

    if(userAns === q.a) {
        quizState.score++;
        document.getElementById('score').innerText = quizState.score;
    }
    
    document.getElementById('explanation').classList.remove('hidden');
    document.getElementById('submit-btn').classList.add('hidden');
    document.getElementById('next-btn').classList.remove('hidden');
}

function nextQuestion() { quizState.currentIdx++; renderQuestion(); }

function showSummary() {
    const pct = Math.round((quizState.score / quizState.questions.length) * 100);
    document.getElementById('quiz-card-container').innerHTML = `
        <div class="slide" style="text-align: center;">
            <h2>Exam Simulation Complete</h2>
            <div style="font-size: 4rem; color: var(--primary); margin: 20px 0; font-weight:bold;">${pct}%</div>
            <p>Score: ${quizState.score} / ${quizState.questions.length}</p>
            <button class="btn-primary" onclick="initQuiz()">Restart Quiz</button>
        </div>`;
    document.getElementById('quiz-footer').classList.add('hidden');
}