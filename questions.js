// Database med 60+ spørsmål
window.rawQuestions = [
    // --- MODULE 1: Intro ---
    { q: "What is the primary distinction between Machine Learning and traditional programming?", options: ["ML is faster", "ML learns rules from data", "ML uses Python", "ML creates databases"], a: 1, exp: "Traditional programming uses explicit hard-coded rules; ML infers rules from data." },
    { q: "Which type of learning deals with labeled data (Input-Output pairs)?", options: ["Unsupervised", "Reinforcement", "Supervised", "Clustering"], a: 2, exp: "Supervised learning requires ground truth labels (e.g., Regression, Classification)." },
    { q: "What is 'NumPy' primarily used for?", options: ["Web dev", "N-dimensional arrays & numerical math", "Visualization", "NLP"], a: 1, exp: "NumPy provides high-performance array objects essential for AI." },
    { q: "Which Scikit-learn function splits data into training and validation sets?", options: ["divide_data()", "train_test_split()", "make_groups()", "split_val()"], a: 1, exp: "train_test_split() is the standard function." },
    { q: "What characterizes Reinforcement Learning?", options: ["Clustering", "Agent maximizes reward", "Predicting prices", "Text processing"], a: 1, exp: "An agent learns by interacting with an environment to get rewards." },
    { q: "Which is an example of Unsupervised Learning?", options: ["Spam Detection", "House Price Prediction", "Customer Segmentation", "Digit Recognition"], a: 2, exp: "Segmentation finds patterns without pre-existing labels." },
    { q: "What does Pandas provide?", options: ["Neural Layers", "DataFrames for data manipulation", "Game engine", "Compilers"], a: 1, exp: "Pandas is the standard for tabular data analysis." },
    { q: "Which library is used for scientific computing and optimization?", options: ["SciPy", "React", "Flask", "Django"], a: 0, exp: "SciPy builds on NumPy for scientific algorithms." },

    // --- MODULE 2: Math ---
    { q: "What does a Vector represent?", options: ["A single number", "A feature array (magnitude/direction)", "A 2D table", "A timestamp"], a: 1, exp: "Vectors are 1D arrays representing features." },
    { q: "If the Dot Product of two vectors is 0, they are:", options: ["Parallel", "Identical", "Orthogonal", "Opposite"], a: 2, exp: "Zero dot product implies 90-degree angle (Uncorrelated)." },
    { q: "What is the downside of One-Hot Encoding?", options: ["Creates sparse vectors", "Too simple", "Cannot represent numbers", "Lossy"], a: 0, exp: "It creates high-dimensional vectors with mostly zeros." },
    { q: "What does PCA maximize?", options: ["Error", "Variance", "Dimensions", "Bias"], a: 1, exp: "PCA preserves the direction of maximum spread (variance)." },
    { q: "What are Eigenvectors in PCA?", options: ["Outliers", "Direction of data spread", "The magnitude", "Labels"], a: 1, exp: "They define the axes of the new feature space." },
    { q: "What is Singular Value Decomposition (SVD)?", options: ["Sorting", "Matrix factorization", "Neural Net", "Clustering"], a: 1, exp: "Decomposes a matrix into three parts; used in PCA." },
    { q: "A square matrix has:", options: ["More rows", "More cols", "Equal rows and cols", "No rows"], a: 2, exp: "Dimensions N x N." },
    { q: "What are Word Embeddings?", options: ["Sparse lists", "Dense vectors capturing meaning", "Random numbers", "One-hot vectors"], a: 1, exp: "They map similar words to nearby points in vector space." },

    // --- MODULE 3: Algorithms ---
    { q: "Cost Function for Linear Regression?", options: ["Accuracy", "MSE (Mean Squared Error)", "Gini", "Entropy"], a: 1, exp: "Measures average squared difference between prediction and actual." },
    { q: "What does 'K' in KNN represent?", options: ["Clusters", "Epochs", "Neighbors to vote", "Learning Rate"], a: 2, exp: "Number of nearest neighbors that determine the class." },
    { q: "Why is KNN a 'Lazy Learner'?", options: ["It's slow", "It doesn't train a model, just stores data", "Low accuracy", "Simple math"], a: 1, exp: "Computation happens only at prediction time." },
    { q: "Goal of SVM?", options: ["Max margin hyperplane", "Shortest path", "Group items", "Predict prob"], a: 0, exp: "Maximize distance between classes." },
    { q: "What is the 'Kernel Trick'?", options: ["Speed up", "Handle non-linear data", "Reduce size", "Remove outliers"], a: 1, exp: "Maps data to higher dimensions to make it separable." },
    { q: "Entropy measures:", options: ["Accuracy", "Impurity/Randomness", "Distance", "Time"], a: 1, exp: "Decision trees split to reduce entropy." },
    { q: "What is a Confusion Matrix?", options: ["Puzzle", "Table of TP/TN/FP/FN", "Neural layer", "Data tool"], a: 1, exp: "Evaluates classification performance." },
    { q: "Recall formula?", options: ["TP/(TP+FP)", "TP/(TP+FN)", "(TP+TN)/Total", "TP/Total"], a: 1, exp: "Ability to find all actual positives." },
    { q: "Critical metric for Cancer detection?", options: ["Precision", "Recall", "Accuracy", "Specificity"], a: 1, exp: "We cannot afford to miss a positive case (False Negative)." },
    { q: "K-Means updates centroids to:", options: ["Random spots", "Mean of assigned points", "Origin", "Edges"], a: 1, exp: "Moves centroid to the center of its cluster." },
    { q: "What is Overfitting?", options: ["Too simple", "Learns noise/training details too well", "Low accuracy", "Too slow"], a: 1, exp: "Model fits training data perfectly but fails on new data." },

    // --- MODULE 4: ANN ---
    { q: "What is a Perceptron?", options: ["Brain", "Math model of a neuron", "Memory", "Cluster"], a: 1, exp: "Inputs * Weights + Bias -> Activation." },
    { q: "Why use Activation Functions?", options: ["Non-linearity", "Convert to text", "Init weights", "Calc error"], a: 0, exp: "Without them, ANN is just linear regression." },
    { q: "Sigmoid range?", options: ["(-1, 1)", "(0, 1)", "(0, inf)", "(-inf, inf)"], a: 1, exp: "Good for probability." },
    { q: "Advantage of ReLU?", options: ["Slower", "Solves Vanishing Gradient", "Negative output", "Complex"], a: 1, exp: "Does not saturate for positive values." },
    { q: "What is Backpropagation?", options: ["Forward pass", "Calc gradients to update weights", "Testing", "Init"], a: 1, exp: "Propagates error backward to adjust weights." },
    { q: "What is an Epoch?", options: ["One batch", "One full dataset pass", "Learning rate", "Layer"], a: 1, exp: "Model has seen all data once." },
    { q: "What is a Batch?", options: ["Full data", "Subset processed before update", "Error", "Neuron"], a: 1, exp: "Subset of data for one gradient update." },
    { q: "Role of Output Layer?", options: ["Features", "Final prediction", "Cleaning", "Init"], a: 1, exp: "Produces the final result." },
    { q: "Learning Rate controls:", options: ["CPU speed", "Step size of weight updates", "Accuracy", "Neurons"], a: 1, exp: "How big a step we take during Gradient Descent." },

    // --- MODULE 5: Deep Learning ---
    { q: "Deep Learning vs Shallow?", options: ["Python", "Multiple hidden layers", "Labeled data", "Unsupervised"], a: 1, exp: "Deep Neural Networks extract features automatically." },
    { q: "CNNs are best for:", options: ["Text", "Images (Grid data)", "Finance", "Clustering"], a: 1, exp: "Preserves spatial structure." },
    { q: "Convolution Layer purpose?", options: ["Reduce size", "Extract features (filters)", "Flatten", "Predict"], a: 1, exp: "Detects edges, shapes, textures." },
    { q: "Pooling Layer purpose?", options: ["Increase params", "Down-sample (reduce size)", "Color", "Invert"], a: 1, exp: "Reduces dimensions and computation." },
    { q: "Problem with standard RNN?", options: ["Too fast", "Vanishing Gradient (Short memory)", "Too deep", "Text only"], a: 1, exp: "Forgets early inputs in long sequences." },
    { q: "How does LSTM solve RNN issues?", options: ["Convolution", "Gates (Input/Output/Forget)", "Small weights", "Sigmoid"], a: 1, exp: "Regulates memory flow." },
    { q: "Autoencoder goal?", options: ["Classification", "Compression/Reconstruction", "Reinforcement", "Regression"], a: 1, exp: "Unsupervised learning of efficient codings." },
    { q: "What is the Bottleneck?", options: ["Input", "Compressed latent space", "Output", "Loss"], a: 1, exp: "Forces the model to learn compression." },
    { q: "Flatten layer does what?", options: ["Smooths", "2D map to 1D vector", "Noise removal", "Cluster"], a: 1, exp: "Prepares for Fully Connected layers." },
    { q: "Tensor in TensorFlow?", options: ["Flowchart", "Multi-dimensional array", "Network", "CPU"], a: 1, exp: "Generalized matrix (N-dimensions)." },

    // --- MODULE 6: NLP ---
    { q: "Tokenization is:", options: ["Translation", "Splitting text to words", "Sentiment", "Grammar"], a: 1, exp: "Breaking text into units." },
    { q: "Stemming is:", options: ["Removing stops", "Cutting word to root", "Counting", "Vectorizing"], a: 1, exp: "Removing suffixes (Fishing->Fish)." },
    { q: "Stop Words are:", options: ["Rare", "Common words (the, a) removed", "Verbs", "Nouns"], a: 1, exp: "Carry little semantic meaning." },
    { q: "TF-IDF measures:", options: ["Word count", "Importance (Freq in doc vs corpus)", "Length", "Grammar"], a: 1, exp: "Highlights unique keywords." },
    { q: "Word2Vec is:", options: ["List", "Dense vector embedding", "Sparse vector", "Database"], a: 1, exp: "Semantic vector representation." },
    { q: "Transformer mechanism?", options: ["Convolution", "Self-Attention", "Recurrence", "Pooling"], a: 1, exp: "Weighs importance of all words at once." },
    { q: "Hugging Face is:", options: ["Robot", "Platform for NLP models", "Algorithm", "Language"], a: 1, exp: "Hub for Transformers." },
    { q: "Transfer Learning is:", options: ["Copying", "Fine-tuning pre-trained model", "Translating", "Scratch"], a: 1, exp: "Using learned knowledge on new task." },
    { q: "Bag of Words:", options: ["Shopping list", "Word frequency (no order)", "Deep model", "Token"], a: 1, exp: "Ignores grammar/order." },
    { q: "Sentiment Analysis:", options: ["Next word", "Classifying emotion", "Summary", "Translation"], a: 1, exp: "Positive/Negative classification." }
];