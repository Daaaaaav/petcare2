import faiss
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

# Load Sentence Transformer Model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS Index Setup (Cosine Similarity)
vector_dim = 384  # Embedding size for MiniLM-L6-v2
faiss_index = faiss.IndexFlatIP(vector_dim)  # Inner Product index for cosine similarity

qa_data = []  # Store (question, answer) pairs

# Function to Normalize Vectors for Cosine Similarity
def normalize_vector(vec):
    return vec / np.linalg.norm(vec)

# Load CSV Files into FAISS Index
def load_csv_to_faiss(csv_files):
    global faiss_index, qa_data
    qa_data.clear()
    faiss_index = faiss.IndexFlatIP(vector_dim)  # Reset FAISS index

    for file_path in csv_files:
        df = pd.read_csv(file_path)

        for _, row in df.iterrows():
            question, answer = row["question"].strip(), row["answer"].strip()
            qa_data.append((question, answer))

            # Encode and Normalize
            embedding = embedder.encode([question])
            embedding = normalize_vector(embedding)  # Normalize for cosine similarity
            faiss_index.add(np.array(embedding, dtype=np.float32))

# Load Data from CSV Files
csv_files = ["Dog-Cat-QA.csv", "Pet-QA.csv"]
load_csv_to_faiss(csv_files)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route("/get_text_response", methods=["POST"])
def get_text_response():
    data = request.json
    question = data.get("message", "").strip().lower()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # 🔹 Predefined Responses for Common Pet Queries
    static_responses = {
        "hello": "Hello! How can I help you with pet care today?",
        "hi": "Hi there! Ask me anything about pet care.",
        "goodbye": "Goodbye! Take care of your pets!",
        "bye": "Bye! Have a great day with your pets!",
        "how are you": "I'm here to help with pet care questions! What do you need assistance with?",

        # Dog Breeds
        "what are different dog breeds": "Some common dog breeds include Labrador Retriever, German Shepherd, Golden Retriever, Bulldog, Beagle, Poodle, Dachshund, and Boxer.",
        "list some dog breeds": "Popular dog breeds include Labrador Retriever, Golden Retriever, Poodle, Beagle, Bulldog, Rottweiler, Boxer, and Doberman.",
        
        # Cat Breeds
        "what are different cat breeds": "Some common cat breeds include Persian, Maine Coon, Siamese, Bengal, Ragdoll, Sphynx, Scottish Fold, and British Shorthair.",
        "list some cat breeds": "Popular cat breeds include Siamese, Maine Coon, Ragdoll, Bengal, Abyssinian, and Persian.",
        
        # Dog Care
        "how often should i bathe my dog": "Most dogs only need a bath every 4-6 weeks unless they get dirty or have skin conditions that require more frequent washing.",
        "what is the best dog food": "The best dog food depends on your dog's breed, age, and health. High-quality brands like Blue Buffalo, Royal Canin, and Hill’s Science Diet are often recommended.",
        
        # Cat Care
        "how often should i clean my cat's litter box": "It’s best to scoop the litter box daily and completely change the litter every 1-2 weeks to keep it clean and odor-free.",
        "what is the best cat food": "High-quality cat food brands include Royal Canin, Blue Buffalo, and Purina Pro Plan. Choose a food suited to your cat’s age and health needs.",
        
        # General Pet Care
        "how do i introduce a new pet to my home": "Introduce a new pet gradually, provide a safe space, and allow them to adjust at their own pace. Supervise initial interactions with other pets.",
        "how can i train my pet": "Positive reinforcement, consistent training, and patience are key. Reward good behavior with treats and avoid punishment-based training.",
    }

    # 🔹 Check for Static Response First
    for key, value in static_responses.items():
        if key in question:
            return jsonify({"response": value})

    # 🔹 Encode and Normalize User Question
    question_embedding = embedder.encode([question])
    question_vector = normalize_vector(np.array(question_embedding, dtype=np.float32))

    # 🔹 Retrieve Top 3 Matches from FAISS
    top_k = 3  
    distances, indices = faiss_index.search(question_vector, top_k)

    # 🔹 Adjusted Similarity Threshold
    similarity_threshold = 0.75  

    matched_answers = []
    for i in range(top_k):
        if distances[0][i] > similarity_threshold:
            matched_answers.append(qa_data[indices[0][i]])

    # 🔹 Debugging: Print FAISS results
    print(f"User Question: {question}")
    for i, (q, a) in enumerate(matched_answers):
        print(f"Match {i + 1}: {q} → {a} (Score: {distances[0][i]})")

    # 🔹 Improved Context Filtering
    keywords_to_category = {
        "breed": ["breed", "dog breeds", "cat breeds", "list breeds"],
        "diet": ["food", "nutrition", "feeding", "diet"],
        "health": ["sick", "illness", "disease", "infection"],
        "care": ["care", "train", "introduce", "clean", "bathe"],
    }

    best_answer = None

    for category, keywords in keywords_to_category.items():
        if any(keyword in question for keyword in keywords):
            for match_question, match_answer in matched_answers:
                if any(keyword in match_question.lower() for keyword in keywords):
                    best_answer = match_answer
                    break
            if best_answer:
                break

    # 🔹 If No Keyword-Based Match, Use Best FAISS Match
    if not best_answer and matched_answers:
        best_answer = matched_answers[0][1]  

    # 🔹 If No Good Match Found, Reject Non-Pet Queries
    if not best_answer:
        return jsonify({"response": "Sorry, I can only answer pet care-related questions."})

    return jsonify({"response": best_answer})


if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    app.run(debug=True, port=5002)