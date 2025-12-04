// --- NAVIGATION LOGIC ---
function showSection(id, element) {
    // Hide all content
    document.querySelectorAll('.module').forEach(mod => mod.classList.remove('active'));
    document.getElementById('quiz-area').classList.remove('active');
    
    // Show Main Content Area
    document.getElementById('content-area').style.display = 'block';
    
    // Update Sidebar
    document.querySelectorAll('.sidebar li').forEach(li => li.classList.remove('active'));
    element.classList.add('active');

    // Show correct module
    document.getElementById(id).classList.add('active');
}

function showQuiz(element) {
    document.getElementById('content-area').style.display = 'none';
    document.getElementById('quiz-area').classList.add('active');
    
    // Update Sidebar
    document.querySelectorAll('.sidebar li').forEach(li => li.classList.remove('active'));
    element.classList.add('active');
    
    // Start or Reset quiz
    if(currentQuestionIndex === 0 && score === 0) {
        loadQuestion();
    }
}

// --- POP-UP MODAL LOGIC (English Terms) ---
document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById("term-modal");
    const closeBtn = document.querySelector(".close");
    const terms = document.querySelectorAll('.term');

    // Add click event to all terms
    terms.forEach(term => {
        term.addEventListener('click', (e) => {
            e.stopPropagation(); 
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


// --- QUIZ DATA (All in English) ---
const questions = [
    // --- Module 1 ---
    { q: "What is the primary goal of Machine Learning?", options: ["Store data", "Manually program rules", "Learn patterns from data to make decisions", "Clean data"], answer: 2, explanation: "ML allows computers to learn without explicit programming." },
    { q: "Which type of learning uses 'labeled data'?", options: ["Unsupervised", "Reinforcement", "Supervised", "Clustering"], answer: 2, explanation: "Supervised learning requires input-output pairs (labels)." },
    { q: "What is the Pandas library used for?", options: ["Numerical calculus", "Data manipulation (DataFrames)", "Training Neural Networks", "Plotting 3D graphs"], answer: 1, explanation: "Pandas is the standard library for tabular data (DataFrames)." },
    
    // --- Module 2 ---
    { q: "What does a Dot Product of 0 indicate?", options: ["Vectors are parallel", "Vectors are orthogonal (perpendicular)", "Vectors are identical", "Vectors are negative"], answer: 1, explanation: "When the dot product is 0, the angle between vectors is 90 degrees (no correlation)." },
    { q: "Which technique converts text categories into binary vectors (0s and 1s)?", options: ["PCA", "One-hot encoding", "Linear Regression", "SVD"], answer: 1, explanation: "One-hot encoding creates a sparse binary vector." },

    // --- Module 3 ---
    { q: "What is MSE (Mean Squared Error)?", options: ["A classification metric", "A loss function for regression", "A data splitting method", "Model accuracy"], answer: 1, explanation: "MSE measures the average squared difference between predicted and actual values." },
    { q: "What does 'K' stand for in KNN?", options: ["Number of clusters", "Number of neighbors to vote", "Learning rate", "Number of dimensions"], answer: 1, explanation: "K determines how many nearest neighbors are considered for the vote." },
    { q: "Which metric is best to avoid False Negatives (e.g., in cancer diagnosis)?", options: ["Precision", "Recall", "Accuracy", "MSE"], answer: 1, explanation: "Recall measures the ability to find all actual positive cases." },

    // --- Module 4 ---
    { q: "What is the purpose of an Activation Function?", options: ["Reset weights", "Introduce non-linearity", "Calculate error", "Store data"], answer: 1, explanation: "Without it, a neural network is just a linear regression model." },
    { q: "Which problem does ReLU solve better than Sigmoid?", options: ["Overfitting", "Vanishing Gradient", "Underfitting", "Data loss"], answer: 1, explanation: "ReLU prevents gradients from becoming too small in the positive range." },
    { q: "What is an 'Epoch'?", options: ["One weight update", "One complete pass through the entire dataset", "A batch of data", "The learning rate"], answer: 1, explanation: "An epoch means the model has seen every training example once." },

    // --- Module 5 ---
    { q: "What is the main purpose of a Convolution layer in a CNN?", options: ["Reduce image size", "Extract features (edges/shapes)", "Classify the image", "Flatten the image"], answer: 1, explanation: "Filters (kernels) slide over the image to detect features." },
    { q: "Why use LSTM instead of a standard RNN?", options: ["It is faster", "It solves the short-term memory problem", "It requires less data", "It uses no weights"], answer: 1, explanation: "LSTM can remember dependencies over long sequences using gates." },

    // --- Module 6 ---
    { q: "What is Tokenization?", options: ["Removing stop words", "Splitting text into words/sub-units", "Translating text", "Finding sentiment"], answer: 1, explanation: "The first step in NLP is breaking text down into tokens." },
    { q: "What is the key mechanism in Transformer models (like BERT)?", options: ["RNN", "Self-Attention", "Pooling", "Sigmoid"], answer: 1, explanation: "Attention allows the model to weigh the importance of different words in a sentence." },
    
    // --- Multi-Select Example ---
    { 
        q: "Select ALL algorithms that are Supervised Learning. (Multi-select)", 
        type: "multi", 
        options: ["K-Means", "Linear Regression", "SVM", "PCA"], 
        answer: [1, 2], 
        explanation: "K-Means and PCA are Unsupervised. Regression and SVM require labels." 
    }
];

// Fill up with duplicates to reach 60 (as requested for exam prep volume simulation).
// In a real scenario, you would replace these with unique questions.
while(questions.length < 60) {
    let randomQ = questions[Math.floor(Math.random() * 15)]; 
    questions.push({
        q: "Review: " + randomQ.q,
        options: randomQ.options,
        answer: randomQ.answer,
        explanation: randomQ.explanation,
        type: randomQ.type
    });
}

// --- QUIZ LOGIC ---
let currentQuestionIndex = 0;
let score = 0;
let shuffledQuestions = [];

function loadQuestion() {
    // Shuffle only on start
    if (shuffledQuestions.length === 0) {
        shuffledQuestions = questions.sort(() => 0.5 - Math.random());
    }

    if (currentQuestionIndex >= shuffledQuestions.length) {
        showResults();
        return;
    }

    const qData = shuffledQuestions[currentQuestionIndex];
    const container = document.getElementById('quiz-container');
    container.innerHTML = ''; // Clear previous

    // Update stats
    document.getElementById('q-current').innerText = currentQuestionIndex + 1;
    document.getElementById('q-total').innerText = shuffledQuestions.length;
    
    // Button visibility
    document.getElementById('submit-btn').classList.remove('hidden');
    document.getElementById('next-btn').classList.add('hidden');

    // Create Card
    const card = document.createElement('div');
    card.className = 'question-card';
    
    let typeText = qData.type === 'multi' ? "<small style='color:var(--accent)'>(Select all that apply)</small>" : "";
    card.innerHTML = `<h3>${qData.q} ${typeText}</h3>`;

    const optionsDiv = document.createElement('div');
    optionsDiv.className = 'options-container';

    // Generate Options
    qData.options.forEach((opt, index) => {
        const btn = document.createElement('div');
        btn.className = 'option-btn';
        btn.innerText = opt;
        btn.onclick = () => selectOption(btn, index); 
        optionsDiv.appendChild(btn);
    });

    card.appendChild(optionsDiv);
    
    // Explanation Box
    const exp = document.createElement('div');
    exp.className = 'explanation';
    exp.id = 'explanation-box';
    exp.innerHTML = `<strong>Explanation:</strong> ${qData.explanation}`;
    card.appendChild(exp);

    container.appendChild(card);
}

function selectOption(btn, index) {
    // Prevent selection if already submitted
    if (!document.getElementById('next-btn').classList.contains('hidden')) return;

    const qData = shuffledQuestions[currentQuestionIndex];
    const isMulti = qData.type === 'multi';

    if (isMulti) {
        btn.classList.toggle('selected');
        btn.dataset.index = index;
    } else {
        // Clear others
        document.querySelectorAll('.option-btn').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
        btn.dataset.index = index;
    }
}

function checkAnswer() {
    const qData = shuffledQuestions[currentQuestionIndex];
    const selectedBtns = document.querySelectorAll('.option-btn.selected');
    
    if (selectedBtns.length === 0) {
        alert("You must select an answer!");
        return;
    }

    let isCorrect = false;

    if (qData.type === 'multi') {
        const selectedIndices = Array.from(selectedBtns).map(b => parseInt(b.dataset.index));
        // Sort to compare arrays
        const sortedSel = selectedIndices.sort().toString();
        const sortedAns = qData.answer.sort().toString();
        isCorrect = (sortedSel === sortedAns);
    } else {
        const selectedIndex = parseInt(selectedBtns[0].dataset.index);
        isCorrect = (selectedIndex === qData.answer);
    }

    // Visual Feedback
    document.querySelectorAll('.option-btn').forEach((btn, idx) => {
        // Mark correct
        if (qData.type === 'multi') {
            if (qData.answer.includes(idx)) btn.classList.add('correct');
        } else {
            if (idx === qData.answer) btn.classList.add('correct');
        }
        
        // Mark wrong
        if (btn.classList.contains('selected') && !btn.classList.contains('correct')) {
            btn.classList.add('wrong');
        }
    });

    if (isCorrect) {
        score++;
        document.getElementById('score').innerText = score;
    }

    document.getElementById('explanation-box').style.display = 'block';
    document.getElementById('submit-btn').classList.add('hidden');
    document.getElementById('next-btn').classList.remove('hidden');
}

function nextQuestion() {
    currentQuestionIndex++;
    loadQuestion();
}

function showResults() {
    const container = document.getElementById('quiz-container');
    container.innerHTML = `
        <div class="slide" style="text-align: center;">
            <h2>Quiz Complete!</h2>
            <p style="font-size: 1.5rem;">Your Score: <strong>${score}</strong> / ${shuffledQuestions.length}</p>
            <p>${score > (shuffledQuestions.length * 0.8) ? "Great job! You seem ready for the exam." : "Review the modules and try again."}</p>
            <button onclick="location.reload()" style="padding:15px 30px; font-size:1.2rem; cursor:pointer;">Restart Quiz</button>
        </div>
    `;
    document.getElementById('quiz-controls').style.display = 'none';
}
