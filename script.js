/* =========================================
   1. NAVIGATION & UI LOGIC
   ========================================= */

function showSection(id, element) {
    document.querySelectorAll('.module').forEach(m => m.classList.remove('active'));
    document.getElementById('quiz-setup').classList.remove('active');
    document.getElementById('quiz-area').classList.remove('active');
    document.getElementById('quiz-area').classList.add('hidden');
    
    const target = document.getElementById(id);
    target.classList.add('active');
    
    document.querySelectorAll('.nav-links li').forEach(l => l.classList.remove('active'));
    element.classList.add('active');
    window.scrollTo(0, 0);
}

function showQuizSetup(element) {
    document.querySelectorAll('.module').forEach(m => m.classList.remove('active'));
    document.getElementById('quiz-area').classList.add('hidden');
    document.getElementById('quiz-setup').classList.add('active');
    
    document.querySelectorAll('.nav-links li').forEach(l => l.classList.remove('active'));
    element.classList.add('active');
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


/* =========================================
   2. QUIZ ENGINE
   ========================================= */

let activeQuestions = [];

function startQuiz(mode) {
    document.getElementById('quiz-setup').classList.remove('active');
    const quizArea = document.getElementById('quiz-area');
    quizArea.classList.remove('hidden');
    quizArea.classList.add('active');
    
    let allQs = window.rawQuestions;
    
    // Filter questions based on mode
    if (mode === 'all') {
        // Shuffle and pick 60 random for Full Exam (Updated from 30)
        activeQuestions = allQs.sort(() => 0.5 - Math.random()).slice(0, 60);
        document.getElementById('quiz-status').innerText = "Full Exam (60 Questions)";
    } else {
        // Filter by specific module
        activeQuestions = allQs.filter(q => q.mod === mode);
        document.getElementById('quiz-status').innerText = `Module ${mode} Practice (${activeQuestions.length} Questions)`;
    }

    renderQuizSheet();
}

function renderQuizSheet() {
    const container = document.getElementById('quiz-list-container');
    container.innerHTML = "";
    document.getElementById('finish-btn').classList.remove('hidden');
    document.getElementById('retry-btn').classList.add('hidden');

    activeQuestions.forEach((q, index) => {
        let html = `
            <div class="quiz-question-block" id="q-block-${index}">
                <div class="quiz-question-text">${index + 1}. ${q.q}</div>
                <div class="quiz-options">
        `;
        
        q.options.forEach((opt, optIdx) => {
            html += `
                <div class="quiz-option" onclick="selectOptionSheet(${index}, ${optIdx}, this)">
                    <div style="width:24px; height:24px; border:2px solid #ccc; border-radius:50%; display:flex; align-items:center; justify-content:center; margin-right:10px;" class="opt-circle"></div>
                    ${opt}
                </div>
            `;
        });

        html += `
                </div>
                <div class="quiz-explanation" id="exp-${index}">
                    <strong><i class="fas fa-info-circle"></i> Explanation:</strong> ${q.exp}
                </div>
            </div>
        `;
        container.innerHTML += html;
    });
    
    window.scrollTo(0,0);
}

function selectOptionSheet(qIdx, optIdx, element) {
    if (document.getElementById('finish-btn').classList.contains('hidden')) return;

    const block = document.getElementById(`q-block-${qIdx}`);
    
    block.querySelectorAll('.quiz-option').forEach(el => {
        el.classList.remove('selected');
        el.querySelector('.opt-circle').style.background = 'transparent';
        el.querySelector('.opt-circle').style.borderColor = '#ccc';
        el.querySelector('.opt-circle').innerText = '';
    });
    
    element.classList.add('selected');
    element.querySelector('.opt-circle').style.background = '#3498db';
    element.querySelector('.opt-circle').style.borderColor = '#3498db';
    
    block.dataset.userAnswer = optIdx;
}

function submitQuizSheet() {
    let score = 0;
    
    activeQuestions.forEach((q, index) => {
        const block = document.getElementById(`q-block-${index}`);
        const userAns = block.dataset.userAnswer;
        const options = block.querySelectorAll('.quiz-option');
        const exp = document.getElementById(`exp-${index}`);

        if (userAns !== undefined) {
            if (parseInt(userAns) === q.a) {
                score++;
                options[userAns].classList.add('correct');
                options[userAns].querySelector('.opt-circle').innerText = '✓';
            } else {
                options[userAns].classList.add('wrong');
                options[userAns].querySelector('.opt-circle').innerText = '✗';
                options[q.a].classList.add('correct'); 
            }
        } else {
             options[q.a].classList.add('correct'); 
        }

        exp.style.display = 'block';
    });

    const percentage = Math.round((score / activeQuestions.length) * 100);
    document.getElementById('quiz-status').innerText = `Score: ${score}/${activeQuestions.length} (${percentage}%)`;
    
    document.getElementById('finish-btn').classList.add('hidden');
    document.getElementById('retry-btn').classList.remove('hidden');
    
    window.scrollTo(0, 0);
}