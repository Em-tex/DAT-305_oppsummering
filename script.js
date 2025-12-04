/* =========================================
   1. NAVIGATION & UI LOGIC
   ========================================= */

function showSection(id, element) {
    document.querySelectorAll('.module').forEach(mod => {
        mod.classList.remove('active');
        mod.style.display = 'none'; 
    });
    
    const quizArea = document.getElementById('quiz-area');
    quizArea.classList.remove('active');
    quizArea.classList.add('hidden'); 
    
    document.getElementById('content-area').style.display = 'block';
    
    const selectedMod = document.getElementById(id);
    selectedMod.style.display = 'block';
    // Small timeout to allow display block to apply before opacity transition
    setTimeout(() => selectedMod.classList.add('active'), 10);

    document.querySelectorAll('.nav-links li').forEach(li => li.classList.remove('active'));
    element.classList.add('active');

    window.scrollTo(0, 0);
}

function showQuiz(element) {
    document.querySelectorAll('.module').forEach(mod => {
        mod.classList.remove('active');
        mod.style.display = 'none';
    });

    const quizArea = document.getElementById('quiz-area');
    quizArea.classList.remove('hidden');
    quizArea.style.display = 'block';
    setTimeout(() => quizArea.classList.add('active'), 10);

    document.querySelectorAll('.nav-links li').forEach(li => li.classList.remove('active'));
    element.classList.add('active');
    
    initQuiz();
}


/* =========================================
   2. QUIZ ENGINE
   ========================================= */

let quizState = {
    questions: [],
    currentIdx: 0,
    score: 0
};

function initQuiz() {
    if (!window.rawQuestions || window.rawQuestions.length === 0) {
        document.getElementById('quiz-card-container').innerHTML = "<p style='color:red; text-align:center;'>Error: Questions not found in questions.js</p>";
        return;
    }

    // Shuffle and pick 60 questions
    let qList = [...window.rawQuestions].sort(() => 0.5 - Math.random());
    quizState.questions = qList.slice(0, 60); 
    quizState.currentIdx = 0;
    quizState.score = 0;

    document.getElementById('quiz-footer').classList.remove('hidden');
    renderQuestion();
}

function renderQuestion() {
    if(quizState.currentIdx >= quizState.questions.length) {
        showSummary();
        return;
    }

    const q = quizState.questions[quizState.currentIdx];
    
    document.getElementById('q-current').innerText = quizState.currentIdx + 1;
    document.getElementById('q-total').innerText = quizState.questions.length;
    document.getElementById('score').innerText = quizState.score;

    document.getElementById('submit-btn').classList.remove('hidden');
    document.getElementById('next-btn').classList.add('hidden');

    let html = `
        <div class="question-card">
            <h2 style="color:var(--primary); margin-bottom:25px; font-size:1.4rem;">${q.q}</h2>
            <div class="options-list">
    `;
    
    q.options.forEach((opt, idx) => {
        html += `
            <div class="option-btn" onclick="selectOption(this, ${idx})">
                <div class="option-icon"></div>
                ${opt}
            </div>`;
    });

    html += `
            </div>
            <div class="explanation hidden" id="explanation">
                <strong><i class="fas fa-info-circle"></i> Explanation:</strong><br> ${q.exp}
            </div>
        </div>
    `;

    document.getElementById('quiz-card-container').innerHTML = html;
}

function selectOption(btn, idx) {
    if(!document.getElementById('next-btn').classList.contains('hidden')) return; 

    document.querySelectorAll('.option-btn').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    btn.dataset.idx = idx;
}

function checkAnswer() {
    const selected = document.querySelector('.option-btn.selected');
    if(!selected) { alert("Please select an option first!"); return; }

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
    
    document.getElementById('score').innerText = quizState.score;
}

function nextQuestion() {
    quizState.currentIdx++;
    renderQuestion();
}

function showSummary() {
    const pct = Math.round((quizState.score / quizState.questions.length) * 100);
    let msg = pct > 80 ? "Outstanding! You are exam ready." : pct > 50 ? "Good job, but review your weak areas." : "Keep studying, you can do this!";
    
    document.getElementById('quiz-card-container').innerHTML = `
        <div class="slide" style="text-align: center; padding: 60px 20px;">
            <i class="fas fa-graduation-cap" style="font-size: 5rem; color: var(--accent); margin-bottom:20px;"></i>
            <h2>Exam Simulation Finished!</h2>
            <div style="font-size: 4rem; color: var(--primary); margin: 20px 0; font-weight:800;">${pct}%</div>
            <p style="font-size:1.2rem;">You scored ${quizState.score} out of ${quizState.questions.length}</p>
            <p style="margin-top:20px; font-weight:bold; color:#7f8c8d;">${msg}</p>
            <button class="btn-primary" style="margin-top:30px;" onclick="initQuiz()">Restart Simulation</button>
        </div>
    `;
    document.getElementById('quiz-footer').classList.add('hidden');
}