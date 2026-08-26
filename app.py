import os
import json
import pickle
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash

from config import Config
from ml.preprocess import clean_text
from ml.responder import generate_suggested_response
from database.db import init_db, add_ticket, get_all_tickets, get_ticket_by_id, delete_ticket, get_ticket_stats

app = Flask(__name__)
app.config.from_object(Config)

# Global variables for loaded ML artifacts
category_model = None
category_vectorizer = None
priority_model = None
priority_vectorizer = None
sentiment_model = None
sentiment_vectorizer = None
metrics_data = {}

# Urgency keywords to assist priority scoring
HIGH_PRIORITY_KEYWORDS = {
    "urgent", "urgently", "immediately", "critical", "cannot access", "payment deducted", 
    "account locked", "completely broken", "emergency", "asap", "down", "crash", "stolen", 
    "unauthorized", "security", "bricked", "leak", "fail", "freeze"
}

def load_ml_models():
    global category_model, category_vectorizer, priority_model, priority_vectorizer
    global sentiment_model, sentiment_vectorizer, metrics_data
    
    models_dir = app.config['MODELS_DIR']
    
    # If models are not trained yet, run training pipeline
    metrics_path = os.path.join(models_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print("[SupportAI] Model files not found. Initiating automated training...")
        from ml.train import train_all_models
        train_all_models()
        
    try:
        with open(os.path.join(models_dir, "category_model.pkl"), "rb") as f:
            category_model = pickle.load(f)
        with open(os.path.join(models_dir, "category_vectorizer.pkl"), "rb") as f:
            category_vectorizer = pickle.load(f)
            
        with open(os.path.join(models_dir, "priority_model.pkl"), "rb") as f:
            priority_model = pickle.load(f)
        with open(os.path.join(models_dir, "priority_vectorizer.pkl"), "rb") as f:
            priority_vectorizer = pickle.load(f)
            
        with open(os.path.join(models_dir, "sentiment_model.pkl"), "rb") as f:
            sentiment_model = pickle.load(f)
        with open(os.path.join(models_dir, "sentiment_vectorizer.pkl"), "rb") as f:
            sentiment_vectorizer = pickle.load(f)
            
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
                
        print("[SupportAI] All ML Models, Vectorizers, and Evaluation Metrics Loaded Successfully.")
    except Exception as e:
        print(f"[SupportAI ERROR] Failed loading models: {e}")

# Initialize DB and Load ML artifacts on app start
with app.app_context():
    init_db()
    load_ml_models()

def predict_ticket_attributes(raw_text: str):
    """
    Runs text preprocessing and multi-attribute ML inference for:
    1. Category (Logistic Regression / Random Forest)
    2. Priority (ML Classifier + Urgency Keyword heuristic)
    3. Sentiment (TF-IDF Classifier)
    """
    cleaned = clean_text(raw_text)
    
    # 1. Category Prediction
    vec_cat = category_vectorizer.transform([cleaned])
    category = category_model.predict(vec_cat)[0]
    cat_probs = category_model.predict_proba(vec_cat)[0]
    cat_confidence = round(float(max(cat_probs)) * 100, 1)
    
    # 2. Priority Prediction (ML + Urgency Rules)
    vec_prio = priority_vectorizer.transform([cleaned])
    ml_priority = priority_model.predict(vec_prio)[0]
    
    # Check urgency keywords
    lower_text = raw_text.lower()
    has_urgency = any(kw in lower_text for kw in HIGH_PRIORITY_KEYWORDS)
    
    if has_urgency and ml_priority != "High":
        priority = "High"
    else:
        priority = ml_priority
        
    # 3. Sentiment Prediction
    vec_sent = sentiment_vectorizer.transform([cleaned])
    sentiment = sentiment_model.predict(vec_sent)[0]
    sent_probs = sentiment_model.predict_proba(vec_sent)[0]
    sent_confidence = round(float(max(sent_probs)) * 100, 1)
    
    # 4. Suggested Response Generation
    suggested_response = generate_suggested_response(raw_text, category, priority, sentiment)
    
    model_name = metrics_data.get("selected_best_model", "Logistic Regression")
    
    return {
        "category": category,
        "category_confidence": cat_confidence,
        "priority": priority,
        "sentiment": sentiment,
        "sentiment_confidence": sent_confidence,
        "suggested_response": suggested_response,
        "model_used": model_name
    }

# -------------------------------------------------------------
# Web Page Routes
# -------------------------------------------------------------

@app.route("/")
def index():
    """Dashboard Page: Overview statistics, KPI cards, and charts."""
    stats = get_ticket_stats()
    return render_template("index.html", stats=stats, page_title="Dashboard")

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    """Live Ticket Analysis Page."""
    return render_template("analyze.html", page_title="Analyze Ticket")

@app.route("/history")
def history():
    """Ticket History Page with search and filter capabilities."""
    search = request.args.get("search", "")
    category = request.args.get("category", "")
    priority = request.args.get("priority", "")
    sentiment = request.args.get("sentiment", "")
    
    tickets = get_all_tickets(
        limit=150, 
        search=search, 
        category_filter=category, 
        priority_filter=priority, 
        sentiment_filter=sentiment
    )
    return render_template(
        "history.html", 
        tickets=tickets, 
        search=search, 
        category=category, 
        priority=priority, 
        sentiment=sentiment, 
        page_title="Ticket History"
    )

@app.route("/performance")
def performance():
    """Model Performance Page: Logistic Regression vs Random Forest metrics comparison."""
    return render_template("performance.html", metrics=metrics_data, page_title="Model Performance")

# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    API endpoint for classifying tickets and generating responses.
    Expects JSON: { "ticket_text": "..." } or form data.
    """
    if request.is_json:
        data = request.get_json()
        ticket_text = data.get("ticket_text", "").strip()
    else:
        ticket_text = request.form.get("ticket_text", "").strip()
        
    if not ticket_text:
        return jsonify({"success": False, "error": "Please enter a customer support ticket text."}), 400
        
    if len(ticket_text) < 5:
        return jsonify({"success": False, "error": "Ticket text is too short. Please provide at least 5 characters."}), 400

    try:
        result = predict_ticket_attributes(ticket_text)
        
        # Save to SQLite database
        ticket_id = add_ticket(
            ticket_text=ticket_text,
            category=result["category"],
            priority=result["priority"],
            sentiment=result["sentiment"],
            suggested_response=result["suggested_response"],
            model_used=result["model_used"]
        )
        
        result["ticket_id"] = ticket_id
        result["success"] = True
        result["ticket_text"] = ticket_text
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "error": f"Inference error: {str(e)}"}), 500

@app.route("/api/stats")
def api_stats():
    """Returns real-time dashboard analytics data for Chart.js."""
    stats = get_ticket_stats()
    return jsonify(stats)

@app.route("/api/tickets/<int:ticket_id>", methods=["DELETE"])
def api_delete_ticket(ticket_id):
    """Deletes a ticket record by ID."""
    success = delete_ticket(ticket_id)
    if success:
        return jsonify({"success": True, "message": f"Ticket #{ticket_id} deleted."})
    return jsonify({"success": False, "error": "Ticket not found."}), 404

# -------------------------------------------------------------
# Error Handlers
# -------------------------------------------------------------

@app.errorhandler(404)
def page_not_found(e):
    return render_template("base.html", error_message="Page Not Found (404)"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("base.html", error_message="Internal Server Error (500)"), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
