window.rawQuestions = [
    // ================= MODULE 1: Intro =================
    { 
        q: "Which statement best describes the difference between Machine Learning (ML) and Traditional Programming?", 
        options: ["ML is always slower than traditional code.", "Traditional programming creates rules from data; ML uses hard-coded rules.", "ML infers rules from data (Input+Output=Rules), whereas traditional programming requires explicit rules.", "ML does not require any data."], 
        a: 2, 
        exp: "Traditional programming relies on humans writing explicit rules. ML algorithms learn these rules by finding patterns in data." 
    },
    { 
        q: "You are given a dataset of emails labeled as 'Spam' or 'Not Spam'. What type of learning task is this?", 
        options: ["Unsupervised Learning", "Reinforcement Learning", "Supervised Learning", "Clustering"], 
        a: 2, 
        exp: "Since the data is labeled (we know the correct answer), it is Supervised Learning. Specifically, a Classification task." 
    },
    { 
        q: "Which library is the foundation for numerical computing in Python (N-dimensional arrays)?", 
        options: ["Pandas", "NumPy", "Matplotlib", "Scikit-Learn"], 
        a: 1, 
        exp: "NumPy (Numerical Python) provides the high-performance array objects that other libraries like Pandas and Scikit-Learn build upon." 
    },
    { 
        q: "Which of the following is an example of Unsupervised Learning?", 
        options: ["Predicting house prices based on size.", "Classifying images of cats and dogs.", "Customer Segmentation (grouping similar customers).", "Teaching a robot to walk."], 
        a: 2, 
        exp: "Customer segmentation typically uses Clustering (like K-Means) to find hidden groups in unlabeled data." 
    },
    { 
        q: "In Reinforcement Learning, what guides the agent's learning?", 
        options: ["Labeled examples", "The reduction of variance", "Reward and Punishment signals", "The covariance matrix"], 
        a: 2, 
        exp: "An agent learns by interacting with an environment and receiving feedback in the form of rewards or punishments." 
    },
    { 
        q: "What is Pandas primarily used for?", 
        options: ["Training Neural Networks", "Data manipulation and analysis (DataFrames)", "Creating 3D games", "Solving complex integrals"], 
        a: 1, 
        exp: "Pandas is the industry standard for handling tabular data (CSVs, Excel) via DataFrames." 
    },
    { 
        q: "Which Scikit-learn function is essential to prevent overfitting during evaluation?", 
        options: ["train_test_split()", "fit()", "predict()", "normalize()"], 
        a: 0, 
        exp: "train_test_split() separates data so we can validate the model on unseen data (test set) rather than the data it learned from." 
    },
    { 
        q: "Deep Learning is a subset of...", 
        options: ["Data Science only", "Machine Learning", "Robotics only", "Web Development"], 
        a: 1, 
        exp: "AI > Machine Learning > Deep Learning. DL is a specialized part of ML using multi-layered neural networks." 
    },
    { 
        q: "What is a 'Label' in the context of Supervised Learning?", 
        options: ["The input feature", "The output or target variable (Y)", "The name of the algorithm", "The noise in the data"], 
        a: 1, 
        exp: "The Label (Target) is the 'answer key' the model tries to learn to predict." 
    },
    { 
        q: "Which library would you use to create a Neural Network?", 
        options: ["NumPy", "Pandas", "Matplotlib", "TensorFlow / Keras"], 
        a: 3, 
        exp: "TensorFlow and Keras are dedicated frameworks for building and training Deep Learning models." 
    },

    // ================= MODULE 2: Math =================
    { 
        q: "If the Dot Product of two vectors is zero, what does this imply?", 
        options: ["They are identical.", "They are orthogonal (perpendicular/uncorrelated).", "They are parallel.", "One vector is zero."], 
        a: 1, 
        exp: "A dot product of 0 means the angle between vectors is 90 degrees, implying no correlation." 
    },
    { 
        q: "What is the shape of a matrix representing a 100x100 pixel RGB color image?", 
        options: ["100x100", "100x100x3", "300x100", "10000x1"], 
        a: 1, 
        exp: "Height x Width x Channels. RGB has 3 channels (Red, Green, Blue)." 
    },
    { 
        q: "What is the primary goal of PCA (Principal Component Analysis)?", 
        options: ["To increase the number of features.", "To classify data.", "To reduce dimensionality while preserving variance.", "To remove all outliers."], 
        a: 2, 
        exp: "PCA simplifies data by projecting it onto fewer axes (Principal Components) that capture the most information (variance)." 
    },
    { 
        q: "In PCA, what does an Eigenvalue represent?", 
        options: ["The direction of the new axis.", "The magnitude (amount of variance) in that direction.", "The error rate.", "The mean of the data."], 
        a: 1, 
        exp: "The Eigenvector is the direction; the Eigenvalue is the magnitude (variance) explained by that direction." 
    },
    { 
        q: "What is One-Hot Encoding used for?", 
        options: ["Compressing images.", "Converting categorical text data into binary vectors.", "Normalizing numbers.", "Creating dense embeddings."], 
        a: 1, 
        exp: "It converts categories (e.g., 'Red', 'Blue') into vectors like [1,0] and [0,1]." 
    },
    { 
        q: "Why is One-Hot Encoding inefficient for large vocabularies?", 
        options: ["It is too slow.", "It creates very sparse, high-dimensional vectors.", "It cannot handle numbers.", "It is inaccurate."], 
        a: 1, 
        exp: "If you have 50,000 words, every word is a vector of size 50,000 with only a single '1'. This wastes memory." 
    },
    { 
        q: "Which matrix factorization method is often used to solve PCA?", 
        options: ["SVD (Singular Value Decomposition)", "LU Decomposition", "Gradient Descent", "Backpropagation"], 
        a: 0, 
        exp: "SVD decomposes a matrix into U, Sigma, and V^T, which reveals the principal components." 
    },
    { 
        q: "What is a 'scalar' in linear algebra?", 
        options: ["A list of numbers", "A 2D grid", "A single number", "A function"], 
        a: 2, 
        exp: "A scalar has magnitude but no direction (just a single value), unlike a vector or matrix." 
    },
    { 
        q: "If you flatten a 28x28 image, what is the size of the resulting vector?", 
        options: ["28", "56", "784", "3"], 
        a: 2, 
        exp: "28 * 28 = 784. Flattening converts a 2D matrix into a 1D vector." 
    },
    { 
        q: "To standardize data before PCA, we typically...", 
        options: ["Multiply by 2.", "Subtract the mean and divide by standard deviation.", "Square all values.", "Remove all negative numbers."], 
        a: 1, 
        exp: "Standardization (Z-score normalization) ensures all features contribute equally to the distance calculations." 
    },

    // ================= MODULE 3: ML Algorithms =================
    { 
        q: "Which metric is the standard Cost Function for Linear Regression?", 
        options: ["Accuracy", "MSE (Mean Squared Error)", "Entropy", "Gini Impurity"], 
        a: 1, 
        exp: "Regression tries to minimize the distance between the line and points. MSE measures the average squared distance." 
    },
    { 
        q: "Why is KNN called a 'Lazy Learner'?", 
        options: ["It is slow.", "It does not learn a model during training; it just stores data.", "It only works on small data.", "It requires no math."], 
        a: 1, 
        exp: "KNN waits until a prediction is requested to calculate distances. There is no training phase." 
    },
    { 
        q: "In SVM, what are 'Support Vectors'?", 
        options: ["The center of the cluster.", "The data points closest to the hyperplane/boundary.", "The outliers.", "The labels."], 
        a: 1, 
        exp: "They are the critical points that define the margin. If you remove other points, the boundary doesn't change." 
    },
    { 
        q: "What is the 'Kernel Trick' in SVM?", 
        options: ["A way to speed up training.", "Mapping non-linear data to a higher dimension to make it linearly separable.", "Using a CPU kernel.", "Removing noise."], 
        a: 1, 
        exp: "It allows SVM to draw linear boundaries in high-dimensional space that appear as non-linear curves in the original space." 
    },
    { 
        q: "In Decision Trees, what does 'Entropy' measure?", 
        options: ["Accuracy", "Impurity or Randomness", "Distance", "Time"], 
        a: 1, 
        exp: "High entropy means a mix of classes (impure). Low entropy means mostly one class (pure). Trees split to reduce entropy." 
    },
    { 
        q: "If your model predicts 'Cancer' but the patient is actually healthy, what type of error is this?", 
        options: ["True Positive", "True Negative", "False Positive", "False Negative"], 
        a: 2, 
        exp: "You predicted Positive (Cancer), but it was False. This is a False Positive (Type I error / False Alarm)." 
    },
    { 
        q: "Which metric is most critical if you cannot afford to miss a positive case (e.g., detecting a fire)?", 
        options: ["Precision", "Accuracy", "Recall (Sensitivity)", "Specificity"], 
        a: 2, 
        exp: "Recall measures how many of the *actual* positives you caught. We want high Recall for safety-critical tasks." 
    },
    { 
        q: "How does K-Means clustering determine which cluster a point belongs to?", 
        options: ["By voting.", "By distance to the nearest centroid.", "By entropy.", "By random assignment."], 
        a: 1, 
        exp: "K-Means assigns every point to the centroid (center) it is geometrically closest to." 
    },
    { 
        q: "What does an AUC of 0.5 indicate?", 
        options: ["Perfect classification.", "The model is guessing randomly.", "The model is broken.", "100% Accuracy."], 
        a: 1, 
        exp: "The ROC curve is a straight diagonal line. The model has no ability to distinguish between classes." 
    },
    { 
        q: "What is the Elbow Method used for?", 
        options: ["Finding the optimal K in K-Means.", "Tuning Learning Rate.", "Pruning Decision Trees.", "Stopping Neural Networks."], 
        a: 0, 
        exp: "You plot inertia vs K. The 'elbow' point indicates the best balance between compression and accuracy." 
    },

    // ================= MODULE 4: ANN =================
    { 
        q: "What is the mathematical purpose of an Activation Function?", 
        options: ["To initialize weights.", "To introduce non-linearity.", "To calculate error.", "To flatten the input."], 
        a: 1, 
        exp: "Without non-linear activation (like ReLU), a Neural Network is just a stack of linear regressions." 
    },
    { 
        q: "Which activation function is susceptible to the 'Vanishing Gradient' problem?", 
        options: ["ReLU", "Leaky ReLU", "Sigmoid", "Maxout"], 
        a: 2, 
        exp: "Sigmoid squashes values between 0 and 1. Derivatives become tiny at the ends, killing the gradient in deep nets." 
    },
    { 
        q: "What happens during Backpropagation?", 
        options: ["Data flows forward to predict.", "The error is calculated.", "Gradients are calculated and propagated backward to update weights.", "The model is saved."], 
        a: 2, 
        exp: "It uses the Chain Rule to figure out how much each weight contributed to the error." 
    },
    { 
        q: "What is an 'Epoch'?", 
        options: ["One forward pass of one image.", "One full pass of the entire dataset through the network.", "The learning rate.", "A type of layer."], 
        a: 1, 
        exp: "Training usually involves multiple epochs (seeing the dataset multiple times)." 
    },
    { 
        q: "What is the role of the 'Optimizer' (e.g., Adam, SGD)?", 
        options: ["To calculate loss.", "To update weights based on gradients to minimize loss.", "To structure the layers.", "To label data."], 
        a: 1, 
        exp: "Gradient Descent is the strategy; the Optimizer is the specific algorithm implementing the update rule." 
    },
    { 
        q: "Why is ReLU preferred over Sigmoid for hidden layers?", 
        options: ["It outputs probabilities.", "It is computationally faster and avoids vanishing gradient.", "It is always negative.", "It creates circles."], 
        a: 1, 
        exp: "ReLU is simple math (max(0,x)) and does not saturate for positive values." 
    },
    { 
        q: "What function is typically used in the Output layer for Multi-Class Classification?", 
        options: ["Sigmoid", "ReLU", "Softmax", "Tanh"], 
        a: 2, 
        exp: "Softmax converts raw output scores (logits) into probabilities that sum to 1." 
    },
    { 
        q: "If the Learning Rate is too high, what happens?", 
        options: ["Training is very slow.", "The model overshoots the minimum and may diverge.", "The model stops immediately.", "Accuracy becomes 100%."], 
        a: 1, 
        exp: "Big steps mean you might step right over the valley (minimum) you are trying to reach." 
    },
    { 
        q: "What is a Perceptron?", 
        options: ["A multi-layer network.", "The simplest form of a neural network (one neuron).", "A recurrent layer.", "An unsupervised algorithm."], 
        a: 1, 
        exp: "A Perceptron takes inputs, weights them, adds bias, and applies a step function." 
    },
    { 
        q: "What is 'Batch Size'?", 
        options: ["Total number of images.", "The number of samples processed before updating the model weights.", "The size of the neuron.", "The learning rate."], 
        a: 1, 
        exp: "We don't update weights after every single image (too slow) or the whole dataset (too memory intensive). We use batches." 
    },

    // ================= MODULE 5: Deep Learning =================
    { 
        q: "What is the primary advantage of CNNs over standard MLPs for images?", 
        options: ["They are faster.", "They capture spatial features (local patterns) via filters.", "They don't use weights.", "They use text."], 
        a: 1, 
        exp: "Standard MLPs flatten images and lose spatial info. CNNs look at patches of pixels to find edges and shapes." 
    },
    { 
        q: "What does a Max Pooling layer do?", 
        options: ["Increases image size.", "Down-samples the image by keeping the max value in a window.", "Applies a filter.", "Changes colors."], 
        a: 1, 
        exp: "It reduces dimensionality and computation while keeping the most prominent feature." 
    },
    { 
        q: "Why do standard RNNs struggle with long sequences?", 
        options: ["They are too fast.", "Vanishing Gradient problem causes short-term memory.", "They have too many gates.", "They only read backwards."], 
        a: 1, 
        exp: "Gradients shrink as they go back through time steps, making the model forget early inputs." 
    },
    { 
        q: "Which gate is NOT part of an LSTM?", 
        options: ["Forget Gate", "Input Gate", "Output Gate", "Random Gate"], 
        a: 3, 
        exp: "LSTMs use Forget, Input, and Output gates to regulate the flow of information." 
    },
    { 
        q: "What is the 'Bottleneck' in an Autoencoder?", 
        options: ["The input layer.", "The compressed latent representation between Encoder and Decoder.", "The output layer.", "The activation function."], 
        a: 1, 
        exp: "The bottleneck forces the network to compress the data, learning efficient features." 
    },
    { 
        q: "What is Transfer Learning?", 
        options: ["Copying files.", "Using a pre-trained model on a new, similar task.", "Training from scratch.", "Unsupervised learning."], 
        a: 1, 
        exp: "Taking a model trained on millions of images (like ImageNet) and fine-tuning it for your specific small dataset." 
    },
    { 
        q: "Which layer in a CNN is responsible for extracting features like edges?", 
        options: ["Pooling Layer", "Convolution Layer", "Fully Connected Layer", "Dropout Layer"], 
        a: 1, 
        exp: "Convolution applies filters (kernels) that activate when they see specific patterns." 
    },
    { 
        q: "What is the goal of an Autoencoder?", 
        options: ["Classification", "Reconstruction of the input (Compression/Denoising).", "Prediction of future values.", "Translation."], 
        a: 1, 
        exp: "It tries to output exactly what was input, forcing it to learn a compressed representation in the middle." 
    },
    { 
        q: "LSTM stands for:", 
        options: ["Linear Standard Time Model", "Long Short-Term Memory", "Large Scale Text Model", "Latent Space Time Machine"], 
        a: 1, 
        exp: "Designed to solve the short-term memory issue of RNNs." 
    },
    { 
        q: "Flattening is usually done...", 
        options: ["Before Convolution.", "Between Convolution and Fully Connected layers.", "At the output.", "Never."], 
        a: 1, 
        exp: "We must convert the 2D feature maps from the CNN layers into a 1D vector to feed the final dense classification layers." 
    },

    // ================= MODULE 6: NLP =================
    { 
        q: "What is Tokenization?", 
        options: ["Translating text.", "Splitting text into smaller units (words/sub-words).", "Counting words.", "Removing punctuation."], 
        a: 1, 
        exp: "The first step in NLP. 'I love AI' -> ['I', 'love', 'AI']." 
    },
    { 
        q: "What is the difference between Stemming and Lemmatization?", 
        options: ["Stemming is slower.", "Stemming just chops off ends; Lemmatization uses grammar/dictionary to find the root.", "They are the same.", "Lemmatization is less accurate."], 
        a: 1, 
        exp: "Stemming: 'Better' -> 'Better'. Lemmatization: 'Better' -> 'Good'." 
    },
    { 
        q: "What does TF-IDF highlight?", 
        options: ["Most common words.", "Words that are important to a specific document but rare in the corpus.", "Stop words.", "Short words."], 
        a: 1, 
        exp: "It penalizes words that appear everywhere (like 'the') and boosts unique keywords." 
    },
    { 
        q: "Word2Vec creates embeddings where...", 
        options: ["Words are sorted alphabetically.", "Semantically similar words are close in vector space.", "Words are assigned random numbers.", "Only nouns are kept."], 
        a: 1, 
        exp: "It allows for math like King - Man + Woman = Queen." 
    },
    { 
        q: "What is the key mechanism behind Transformers (BERT/GPT)?", 
        options: ["Convolution", "Self-Attention", "Recurrence", "Max Pooling"], 
        a: 1, 
        exp: "Self-Attention allows the model to weigh the importance of every word relative to every other word in the sequence." 
    },
    { 
        q: "BERT is an ______ model.", 
        options: ["Encoder-only", "Decoder-only", "RNN", "Linear"], 
        a: 0, 
        exp: "BERT (Bidirectional Encoder Representations) uses the Encoder stack to understand text context." 
    },
    { 
        q: "GPT is a ______ model.", 
        options: ["Encoder-only", "Decoder-only (Autoregressive)", "Clustering", "CNN"], 
        a: 1, 
        exp: "GPT (Generative Pre-trained Transformer) uses the Decoder stack to generate the next word." 
    },
    { 
        q: "What are 'Stop Words'?", 
        options: ["Negative words.", "Common words (the, a, is) usually removed during preprocessing.", "The end of a sentence.", "Errors."], 
        a: 1, 
        exp: "They carry little semantic meaning and are often noise." 
    },
    { 
        q: "Hugging Face is famous for...", 
        options: ["Robotics.", "The 'Transformers' library and Model Hub.", "Web design.", "Database management."], 
        a: 1, 
        exp: "It is the central hub for open-source NLP models." 
    },
    { 
        q: "In NLP, what is a 'Bag of Words'?", 
        options: ["A list of all words.", "Representing text by word frequency, ignoring order.", "A deep learning model.", "A dictionary."], 
        a: 1, 
        exp: "It counts how many times words appear but loses context (word order)." 
    }
];