import faiss 
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

# ------------------------------
# Model Comparison Setup
# ------------------------------
# Two models for comparison
embedding_models = {
    "mini": SentenceTransformer("all-MiniLM-L6-v2"),
    "mpnet": SentenceTransformer("all-mpnet-base-v2")
}
# Set the default model
current_model_key = "mini"
current_model = embedding_models[current_model_key]

# Embedding dimensions for each model (they differ!)
model_vector_dims = {
    "mini": 384,
    "mpnet": 768
}

# FAISS indexes for each model and storage for embeddings per model
faiss_indexes = {}
model_embeddings = {model_key: [] for model_key in embedding_models.keys()}

# ------------------------------
# Data Loading & Preprocessing
# ------------------------------
qa_data = []  # store QA pairs

def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm

def load_csv_to_faiss(csv_files):
    global qa_data, faiss_indexes, model_embeddings
    qa_data.clear()
    # Reinitialize embeddings and FAISS indexes for each model
    for key in model_embeddings:
        model_embeddings[key] = []
        dim = model_vector_dims[key]
        faiss_indexes[key] = faiss.IndexFlatIP(dim)

    for file_path in csv_files:
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            question, answer = row["question"].strip(), row["answer"].strip()
            qa_data.append((question, answer))
            # Process each model's embedding
            for model_key, model in embedding_models.items():
                embedding = model.encode([question])
                embedding = normalize_vector(embedding)
                model_embeddings[model_key].append(embedding[0])
                faiss_indexes[model_key].add(np.array([embedding[0]], dtype=np.float32))

# Load your CSV files
csv_files = ["Dog-Cat-QA.csv", "Pet-QA.csv"]
load_csv_to_faiss(csv_files)

# ------------------------------
# User Feedback & Evaluation
# ------------------------------
# Each feedback record now stores:
#   question, answer, rating, predicted (binary), actual (binary), and model used.
feedbacks = []

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    question = data.get("question")
    answer = data.get("answer")
    rating = data.get("rating")  # e.g., 1-5 rating scale
    predicted = data.get("predicted", 1)  # default predicted value is 1 (answer provided)
    actual = data.get("actual")  # user-provided: 1 if correct/helpful, 0 otherwise
    # Also capture which model was used; default to current_model_key
    model_key = data.get("model", current_model_key)
    
    if question and answer and (rating is not None) and (actual is not None):
        try:
            rating = float(rating)
            predicted = int(predicted)
            actual = int(actual)
            if actual not in [0, 1]:
                raise ValueError("Actual must be 0 or 1.")
        except ValueError as e:
            return jsonify({"error": f"Invalid data: {str(e)}"}), 400

        feedbacks.append({
            'model': model_key,
            'question': question,
            'answer': answer,
            'rating': rating,
            'predicted': predicted,
            'actual': actual
        })
        return jsonify({"message": f"Feedback received for model '{model_key}'."})
    else:
        return jsonify({"error": "Missing question, answer, rating, or actual value."}), 400

@app.route("/evaluation", methods=["GET"])
def evaluation():
    if not feedbacks:
        return jsonify({"message": "No feedback available yet."})
    
    # Calculate evaluation metrics for each model separately
    model_metrics = {}
    for model_key in embedding_models.keys():
        model_feedbacks = [f for f in feedbacks if f["model"] == model_key]
        total = len(model_feedbacks)

        if total == 0:
            model_metrics[model_key] = {
                "feedback_count": 0,
                "message": "No feedback for this model."
            }
            continue

        TP = sum(1 for f in model_feedbacks if f['predicted'] == 1 and f['actual'] == 1)
        TN = sum(1 for f in model_feedbacks if f['predicted'] == 0 and f['actual'] == 0)
        FP = sum(1 for f in model_feedbacks if f['predicted'] == 1 and f['actual'] == 0)
        FN = sum(1 for f in model_feedbacks if f['predicted'] == 0 and f['actual'] == 1)

        accuracy = (TP + TN) / total if total else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        model_metrics[model_key] = {
            "feedback_count": total,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

    return jsonify(model_metrics)

# ------------------------------
# Model Switching Endpoint
# ------------------------------
@app.route("/switch_model", methods=["POST"])
def switch_model():
    global current_model_key, current_model
    data = request.json
    model_key = data.get("model")
    if model_key in embedding_models:
        current_model_key = model_key
        current_model = embedding_models[model_key]
        return jsonify({"message": f"Switched to model '{model_key}'."})
    else:
        return jsonify({"error": "Model key not recognized."}), 400

# ------------------------------
# Main Chatbot Response Endpoint
# ------------------------------
@app.route('/')
def index():
    return render_template('chat.html')

@app.route("/get_text_response", methods=["POST"])
def get_text_response():
    data = request.json
    question_input = data.get("message", "").strip().lower()

    if not question_input:
        return jsonify({"error": "No question provided"}), 400

    # Consistent response formatting
    def format_response(resp):
        return f"Chatbot: {resp}"

    # Static responses for common queries
    static_responses = {
        "hello": "Hello! How can I help you with pet care today?",
        "hi": "Hi there! Ask me anything about pet care.",
        "goodbye": "Goodbye! Take care of your pets!",
        "bye": "Bye! Have a great day with your pets!",
        "how are you": "I'm here to help with pet care questions! What do you need assistance with?",
        "what are different dog breeds": "Some common dog breeds include Labrador Retriever, German Shepherd, Golden Retriever, Bulldog, Beagle, Poodle, Dachshund, and Boxer.",
        "list some dog breeds": "Popular dog breeds include Labrador Retriever, Golden Retriever, Poodle, Beagle, Bulldog, Rottweiler, Boxer, and Doberman.",
        "what are different cat breeds": "Some common cat breeds include Persian, Maine Coon, Siamese, Bengal, Ragdoll, Sphynx, Scottish Fold, and British Shorthair.",
        "list some cat breeds": "Popular cat breeds include Siamese, Maine Coon, Ragdoll, Bengal, Abyssinian, and Persian.",
        "how often should i bathe my dog": "Most dogs only need a bath every 4-6 weeks unless they get dirty or have skin conditions that require more frequent washing.",
        "what is the best dog food": "The best dog food depends on your dog's breed, age, and health. High-quality brands like Blue Buffalo, Royal Canin, and Hill’s Science Diet are often recommended.",
        "how often should i clean my cat's litter box": "It’s best to scoop the litter box daily and completely change the litter every 1-2 weeks to keep it clean and odor-free.",
        "what is the best cat food": "High-quality cat food brands include Royal Canin, Blue Buffalo, and Purina Pro Plan. Choose a food suited to your cat’s age and health needs.",
        "how do i introduce a new pet to my home": "Introduce a new pet gradually, provide a safe space, and allow them to adjust at their own pace. Supervise initial interactions with other pets.",
        "how can i train my pet": "Positive reinforcement, consistent training, and patience are key. Reward good behavior with treats and avoid punishment-based training.",
    }

    for key, value in static_responses.items():
        if key in question_input:
            return jsonify({"response": format_response(value)})

    # Use the currently selected model and its FAISS index
    current_faiss_index = faiss_indexes[current_model_key]
    question_embedding = current_model.encode([question_input])
    question_vector = normalize_vector(np.array(question_embedding, dtype=np.float32))

    top_k = 3
    distances, indices = current_faiss_index.search(question_vector, top_k)
    similarity_threshold = 0.75

    matched_answers = []
    for i in range(top_k):
        if distances[0][i] > similarity_threshold:
            matched_answers.append(qa_data[indices[0][i]])

    # Debug logging
    print(f"User Question: {question_input}")
    for i, (q, a) in enumerate(matched_answers):
        print(f"Match {i + 1}: {q} -> {a} (Score: {distances[0][i]})")

    # Select answer based on keyword categories
    keywords_to_category = {
        "breed": ["breed", "dog breeds", "cat breeds", "list breeds"],
        "diet": ["food", "nutrition", "feeding", "diet"],
        "health": ["sick", "illness", "disease", "infection"],
        "care": ["care", "train", "introduce", "clean", "bathe"],
    }

    best_answer = None
    for category, keywords in keywords_to_category.items():
        if any(keyword in question_input for keyword in keywords):
            for match_question, match_answer in matched_answers:
                if any(keyword in match_question.lower() for keyword in keywords):
                    best_answer = match_answer
                    break
            if best_answer:
                break

    if not best_answer and matched_answers:
        best_answer = matched_answers[0][1]  # use the answer from the first match

    if not best_answer:
        return jsonify({"response": format_response("Sorry, I can only answer pet care-related questions.")})

    return jsonify({"response": format_response(best_answer)})

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True, port=5002)
