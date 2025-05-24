import json
import random
from datetime import datetime, timedelta
import uuid

# Sample dataset generator for the Student Engagement Analyzer
def generate_sample_dataset():
    """Generate realistic multi-speaker transcripts for testing"""
    
    # Participants
    host = "Keerthana"
    students = ["Rafi", "Sreejith", "Aayush", "Sarthak", "Devashis", "Sashi Pavan", "Shruti"]
    
    # Session ID
    session_id = str(uuid.uuid4())
    
    # Sample conversation about Machine Learning fundamentals
    conversation_data = [
        {
            "speaker": "Keerthana",
            "text": "Good morning everyone! Today we're going to dive deep into machine learning fundamentals. Let's start with understanding what makes a good machine learning model. The key principle is the bias-variance tradeoff. High bias models are too simplistic and underfit the data, while high variance models are too complex and overfit. The sweet spot is finding the right balance that minimizes both bias and variance to achieve optimal generalization performance.",
            "timestamp": "2024-01-20T10:00:00",
            "duration": 45
        },
        {
            "speaker": "Rafi",
            "text": "Professor, can you explain more about how we identify if our model is overfitting?",
            "timestamp": "2024-01-20T10:00:45",
            "duration": 8
        },
        {
            "speaker": "Keerthana",
            "text": "Excellent question, Rafi! Overfitting typically manifests when your training accuracy is very high, but validation accuracy is significantly lower. We use techniques like cross-validation to detect this. Another telltale sign is when your learning curves show the training error continuing to decrease while validation error starts increasing. This gap indicates your model is memorizing training data rather than learning generalizable patterns. Regularization techniques like L1, L2, dropout, and early stopping help combat overfitting.",
            "timestamp": "2024-01-20T10:00:53",
            "duration": 55
        },
        {
            "speaker": "Sreejith",
            "text": "What's the difference between L1 and L2 regularization in practical terms?",
            "timestamp": "2024-01-20T10:01:48",
            "duration": 12
        },
        {
            "speaker": "Keerthana",
            "text": "Great follow-up question! L1 regularization, also called Lasso, adds the sum of absolute values of parameters to the loss function. This creates sparse models by driving some weights to exactly zero, effectively performing feature selection. L2 regularization, or Ridge regression, adds the sum of squared parameters. It shrinks weights towards zero but rarely makes them exactly zero. L1 is better when you want automatic feature selection and interpretable models with fewer features. L2 works well when all features are somewhat relevant and you want to prevent any single feature from dominating.",
            "timestamp": "2024-01-20T10:02:00",
            "duration": 65
        },
        {
            "speaker": "Aayush",
            "text": "I'm still confused about the mathematical intuition behind why L2 shrinks weights but doesn't make them zero like L1 does.",
            "timestamp": "2024-01-20T10:03:05",
            "duration": 15
        },
        {
            "speaker": "Keerthana",
            "text": "Perfect question for deeper understanding! The mathematical difference lies in the derivative behavior. L1's derivative is constant (+1 or -1), providing consistent pressure to reduce weights by a fixed amount each step, eventually reaching zero. L2's derivative is proportional to the weight value itself (2w), so as weights get smaller, the regularization pressure decreases proportionally. This creates a 'soft landing' effect where weights asymptotically approach but rarely reach exactly zero. Think of L1 as a constant push towards zero, while L2 is like friction that gradually slows down movement towards zero.",
            "timestamp": "2024-01-20T10:03:20",
            "duration": 70
        },
        {
            "speaker": "Sarthak",
            "text": "This makes sense! So if I have a dataset with many irrelevant features, L1 would be better?",
            "timestamp": "2024-01-20T10:04:30",
            "duration": 10
        },
        {
            "speaker": "Keerthana",
            "text": "Exactly right, Sarthak! When you suspect many features are irrelevant or redundant, L1 regularization automatically performs feature selection by zeroing out unimportant weights. This is particularly valuable in high-dimensional datasets where interpretability matters. However, there's also Elastic Net, which combines both L1 and L2 penalties, giving you the benefits of both: feature selection from L1 and the stability of L2. The mixing parameter alpha controls the balance between the two regularization types.",
            "timestamp": "2024-01-20T10:04:40",
            "duration": 50
        },
        {
            "speaker": "Devashis",
            "text": "How do we choose the right regularization parameter? Is it just trial and error?",
            "timestamp": "2024-01-20T10:05:30",
            "duration": 12
        },
        {
            "speaker": "Keerthana",
            "text": "Not just trial and error, Devashis! We use systematic approaches like grid search or random search with cross-validation. The key is to test multiple regularization strength values and select the one that gives the best cross-validation performance. Modern techniques include Bayesian optimization for hyperparameter tuning, which is more efficient than grid search. You can also use learning curves to visualize how different regularization values affect the bias-variance tradeoff. Too little regularization leads to overfitting, too much leads to underfitting.",
            "timestamp": "2024-01-20T10:05:42",
            "duration": 60
        },
        {
            "speaker": "Sashi Pavan",
            "text": "Professor, in real-world projects, how do you decide between different algorithms like Random Forest, SVM, or Neural Networks?",
            "timestamp": "2024-01-20T10:06:42",
            "duration": 18
        },
        {
            "speaker": "Keerthana",
            "text": "Excellent practical question! Algorithm selection depends on several factors: dataset size, feature types, interpretability requirements, and computational constraints. Random Forest works well for tabular data with mixed feature types and provides good feature importance. SVMs excel with high-dimensional data and clear margin separation. Neural networks shine with large datasets, complex patterns, and when you have sufficient computational resources. Start with simpler models like logistic regression or random forest for baseline performance, then move to more complex models if needed. Always consider the 'no free lunch' theorem - no single algorithm works best for all problems.",
            "timestamp": "2024-01-20T10:07:00",
            "duration": 75
        },
        {
            "speaker": "Shruti",
            "text": "What about ensemble methods? When should we use them instead of single models?",
            "timestamp": "2024-01-20T10:08:15",
            "duration": 12
        },
        {
            "speaker": "Keerthana",
            "text": "Great question, Shruti! Ensemble methods combine multiple models to achieve better performance than individual models. Use bagging methods like Random Forest when you want to reduce variance of high-variance models like decision trees. Boosting methods like XGBoost or AdaBoost are excellent when you want to reduce bias by combining weak learners. Stacking allows you to combine different types of algorithms. The key principle is that diverse models with different strengths and weaknesses can complement each other. However, ensembles trade-off interpretability for performance and require more computational resources.",
            "timestamp": "2024-01-20T10:08:27",
            "duration": 68
        },
        {
            "speaker": "Rafi",
            "text": "How do we ensure our models are fair and don't have biased predictions, especially for sensitive applications?",
            "timestamp": "2024-01-20T10:09:35",
            "duration": 15
        },
        {
            "speaker": "Keerthana",
            "text": "Crucial question in today's AI landscape, Rafi! Algorithmic fairness involves multiple considerations: statistical parity, equalized odds, and individual fairness. First, audit your training data for historical biases. Use techniques like demographic parity to ensure equal outcomes across protected groups, or equalized opportunity to ensure equal true positive rates. Tools like IBM's AI Fairness 360 or Google's What-If Tool help detect and mitigate bias. However, remember that fairness is not just a technical problem - it requires domain expertise and stakeholder input to define what fairness means in your specific context.",
            "timestamp": "2024-01-20T10:09:50",
            "duration": 70
        },
        {
            "speaker": "Sreejith",
            "text": "Are there any preprocessing techniques specifically for addressing bias in the data?",
            "timestamp": "2024-01-20T10:11:00",
            "duration": 10
        },
        {
            "speaker": "Keerthana",
            "text": "Absolutely! Pre-processing approaches include re-sampling techniques to balance representation across groups, data augmentation to increase diversity, and feature engineering to remove proxy variables that might encode bias. You can also use adversarial debiasing during training, where you train a model to predict the target while simultaneously training an adversary to detect protected attributes from the model's predictions. The goal is to learn representations that are predictive but don't encode sensitive information.",
            "timestamp": "2024-01-20T10:11:10",
            "duration": 55
        },
        {
            "speaker": "Aayush",
            "text": "What evaluation metrics should we use to measure model performance beyond just accuracy?",
            "timestamp": "2024-01-20T10:12:05",
            "duration": 12
        },
        {
            "speaker": "Keerthana",
            "text": "Essential question for robust model evaluation! Accuracy alone can be misleading, especially with imbalanced datasets. Use precision and recall for understanding false positives and false negatives. F1-score balances both. For multi-class problems, consider macro and micro averaging. AUC-ROC shows performance across different thresholds. For business applications, consider cost-sensitive metrics that reflect real-world impact. Don't forget calibration metrics like Brier score to ensure predicted probabilities are meaningful. Always use stratified cross-validation to ensure robust estimates across different data splits.",
            "timestamp": "2024-01-20T10:12:17",
            "duration": 65
        },
        {
            "speaker": "Sarthak",
            "text": "How do we handle missing data effectively? I've seen different approaches and I'm not sure which one to use.",
            "timestamp": "2024-01-20T10:13:22",
            "duration": 15
        },
        {
            "speaker": "Keerthana",
            "text": "Missing data handling is crucial for model performance! Simple approaches include mean/median imputation for numerical features and mode imputation for categorical features. More sophisticated methods include KNN imputation, which uses similar observations, or iterative imputation like MICE (Multiple Imputation by Chained Equations). For time series, forward-fill or interpolation might be appropriate. However, always investigate why data is missing - it might be informative! Missing Completely at Random (MCAR), Missing at Random (MAR), and Missing Not at Random (MNAR) require different strategies. Sometimes creating a 'missing' indicator feature captures valuable information.",
            "timestamp": "2024-01-20T10:13:37",
            "duration": 80
        },
        {
            "speaker": "Devashis",
            "text": "Can you explain the concept of feature engineering and its importance in machine learning pipelines?",
            "timestamp": "2024-01-20T10:14:57",
            "duration": 14
        },
        {
            "speaker": "Keerthana",
            "text": "Feature engineering is often the difference between good and great models! It involves creating new features from existing ones to better represent the underlying patterns. Examples include polynomial features for capturing non-linear relationships, interaction terms for feature combinations, binning continuous variables, and extracting date/time components. Domain knowledge is crucial here - understanding the problem helps create meaningful features. Automated feature engineering tools like Featuretools can help, but domain expertise remains irreplaceable. Remember the garbage in, garbage out principle - better features often matter more than complex algorithms.",
            "timestamp": "2024-01-20T10:15:11",
            "duration": 70
        },
        {
            "speaker": "Sashi Pavan",
            "text": "What about feature scaling? When do we need it and what are the different methods?",
            "timestamp": "2024-01-20T10:16:21",
            "duration": 12
        },
        {
            "speaker": "Keerthana",
            "text": "Feature scaling is essential for distance-based algorithms like KNN, SVM, and neural networks, but not necessary for tree-based methods. Min-Max scaling transforms features to a fixed range, typically [0,1]. Standardization (Z-score normalization) centers data around zero with unit variance - better when features follow normal distribution. Robust scaling uses median and IQR, making it less sensitive to outliers. Max scaling divides by the maximum absolute value. Choose based on your data distribution and algorithm requirements. Always fit the scaler on training data only, then transform both training and test sets to prevent data leakage.",
            "timestamp": "2024-01-20T10:16:33",
            "duration": 75
        },
        {
            "speaker": "Shruti",
            "text": "How do we handle categorical variables with high cardinality, like user IDs or product codes?",
            "timestamp": "2024-01-20T10:17:48",
            "duration": 13
        },
        {
            "speaker": "Keerthana",
            "text": "High cardinality categorical variables pose unique challenges! One-hot encoding becomes impractical with thousands of categories. Alternative approaches include target encoding (replacing categories with their target mean), but beware of overfitting - use cross-validation for robust estimates. Frequency encoding replaces categories with their occurrence counts. For hierarchical categories, you can create embeddings using techniques from NLP. Hash encoding maps categories to fixed-size buckets. Consider grouping rare categories into an 'other' category. The choice depends on the relationship between the categorical variable and your target variable.",
            "timestamp": "2024-01-20T10:18:01",
            "duration": 70
        },
        {
            "speaker": "Rafi",
            "text": "What are some common pitfalls in machine learning projects that we should avoid?",
            "timestamp": "2024-01-20T10:19:11",
            "duration": 10
        },
        {
            "speaker": "Keerthana",
            "text": "Excellent question to wrap up our discussion! Common pitfalls include data leakage - using future information to predict the past, or including the target variable in features. Poor train-test splits, especially with time series data where you need temporal splits. Survivorship bias in your dataset. Ignoring class imbalance. Over-engineering features without proper validation. Not considering model interpretability requirements early. Inadequate baseline comparisons. Most importantly, losing sight of the business problem you're trying to solve. Always start with simple models, understand your data thoroughly, and iterate based on solid experimental design principles.",
            "timestamp": "2024-01-20T10:19:21",
            "duration": 85
        },
        {
            "speaker": "Sreejith",
            "text": "Thank you Professor! This has been really insightful. I feel much more confident about tackling ML problems now.",
            "timestamp": "2024-01-20T10:20:46",
            "duration": 8
        },
        {
            "speaker": "Aayush",
            "text": "Yes, especially the part about bias-variance tradeoff and regularization techniques. The mathematical intuition really helped.",
            "timestamp": "2024-01-20T10:20:54",
            "duration": 12
        },
        {
            "speaker": "Keerthana",
            "text": "I'm glad you found it helpful! Remember, machine learning is as much about understanding your data and problem domain as it is about algorithms. Keep practicing with real datasets, and don't hesitate to ask questions. For our next session, please review the concepts we discussed today and think about how you might apply them to your course projects.",
            "timestamp": "2024-01-20T10:21:06",
            "duration": 40
        }
    ]
    
    # Convert to the format expected by the API
    formatted_transcripts = []
    for entry in conversation_data:
        formatted_transcripts.append({
            "speaker": entry["speaker"],
            "text": entry["text"],
            "timestamp": entry["timestamp"],
            "session_id": session_id
        })
    
    return {
        "session_id": session_id,
        "transcripts": formatted_transcripts,
        "participants": [host] + students,
        "host": host,
        "total_duration_minutes": 21,
        "topic": "Machine Learning Fundamentals - Bias-Variance Tradeoff and Regularization"
    }

def save_sample_data():
    """Save sample data to JSON file"""
    data = generate_sample_dataset()
    
    with open("sample_dataset.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Sample dataset saved to sample_dataset.json")
    print(f"Session ID: {data['session_id']}")
    print(f"Total transcripts: {len(data['transcripts'])}")
    print(f"Participants: {', '.join(data['participants'])}")
    
    return data

def generate_additional_sessions():
    """Generate multiple sessions for testing"""
    sessions = []
    
    # Session 2: Deep Learning and Neural Networks
    session2_data = [
        {
            "speaker": "Keerthana",
            "text": "Today we'll explore deep learning and neural networks. Neural networks are universal function approximators, capable of learning complex non-linear relationships. The key insight is that by stacking multiple layers of neurons with non-linear activation functions, we can approximate any continuous function. However, this power comes with challenges: vanishing gradients, overfitting, and the need for large datasets.",
            "timestamp": "2024-01-21T10:00:00",
            "duration": 60
        },
        {
            "speaker": "Devashis",
            "text": "What exactly causes the vanishing gradient problem?",
            "timestamp": "2024-01-21T10:01:00",
            "duration": 8
        },
        {
            "speaker": "Keerthana",
            "text": "Great question! Vanishing gradients occur during backpropagation when gradients become exponentially smaller as they propagate backward through layers. This happens because gradients are computed using the chain rule, multiplying many small derivatives together. Traditional activation functions like sigmoid have derivatives that are at most 0.25, so multiplying many of these together results in very small gradients in early layers. This makes it difficult for deep networks to learn effectively.",
            "timestamp": "2024-01-21T10:01:08",
            "duration": 65
        },
        {
            "speaker": "Shruti",
            "text": "How do modern architectures like ResNet solve this problem?",
            "timestamp": "2024-01-21T10:02:13",
            "duration": 10
        }
    ]
    
    # Add more sessions as needed...
    
    return sessions

if __name__ == "__main__":
    # Generate and save sample data
    sample_data = save_sample_data()
    
    # Display some statistics
    print("\n--- Engagement Statistics ---")
    speaker_word_counts = {}
    for transcript in sample_data["transcripts"]:
        speaker = transcript["speaker"]
        word_count = len(transcript["text"].split())
        if speaker not in speaker_word_counts:
            speaker_word_counts[speaker] = 0
        speaker_word_counts[speaker] += word_count
    
    for speaker, word_count in sorted(speaker_word_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{speaker}: {word_count} words")
    
    print(f"\nHost (Keerthana) spoke {speaker_word_counts.get('Keerthana', 0)} words")
    student_words = sum(count for speaker, count in speaker_word_counts.items() if speaker != 'Keerthana')
    print(f"Students collectively spoke {student_words} words")
    
    print("\n--- Sample Poll Questions That Could Be Generated ---")
    print("1. What is the fundamental difference between L1 and L2 regularization in terms of their mathematical derivatives?")
    print("2. In the context of the bias-variance tradeoff, explain why ensemble methods like Random Forest reduce variance.")
    print("3. Given a dataset with many irrelevant features, which regularization technique would be most appropriate and why?")
    print("4. How would you detect and address algorithmic bias in a machine learning model for a sensitive application?")