import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "instance", "supportai.db")

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_text TEXT NOT NULL,
            category TEXT NOT NULL,
            priority TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            suggested_response TEXT NOT NULL,
            model_used TEXT DEFAULT 'Logistic Regression',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    
    # Check if empty; if so, seed sample tickets for immediate presentation demonstration
    cursor.execute("SELECT COUNT(*) FROM tickets")
    count = cursor.fetchone()[0]
    
    if count == 0:
        seed_samples = [
            (
                "My payment was deducted from my account but my order was not placed. I need this resolved urgently.",
                "Billing / Payment",
                "High",
                "Negative",
                "We deeply apologize for the critical inconvenience and frustration this issue has caused.\n\nYour billing inquiry regarding payment transactions or account charges has been registered. Our financial operations team is auditing the transaction details and payment gateway logs. If any unauthorized or duplicate charge is identified, a prompt reversal or adjustment will be initiated.\n\nDue to the urgent nature of this issue (High Priority), your ticket has been expedited for priority investigation within 1-2 hours.\n\nPlease reply directly to this message if you have additional details to add to your case.",
                "Logistic Regression",
                "2026-08-26 10:15:00"
            ),
            (
                "I have been trying to log into my account but my password reset isn't working.",
                "Account / Login",
                "Medium",
                "Negative",
                "We apologize for the inconvenience you are experiencing with our service.\n\nWe understand you are having difficulties accessing your account. Please ensure you are using the latest verification link and check your spam folder for authentication emails.\n\nYour ticket has been queued with standard priority and an agent will follow up within 4-6 business hours.\n\nPlease reply directly to this message if you have additional details to add to your case.",
                "Logistic Regression",
                "2026-08-26 11:30:00"
            ),
            (
                "The web application is throwing 500 internal server error when uploading large files.",
                "Technical Issue",
                "High",
                "Negative",
                "We deeply apologize for the critical inconvenience and frustration this issue has caused.\n\nWe are sorry you are encountering a technical error. Our engineering team has logged the diagnostics.\n\nDue to the urgent nature of this issue (High Priority), your ticket has been expedited for priority investigation within 1-2 hours.\n\nPlease reply directly to this message if you have additional details to add to your case.",
                "Logistic Regression",
                "2026-08-26 12:45:00"
            ),
            (
                "Can you provide a detailed receipt and tax invoice for my recent purchase?",
                "Billing / Payment",
                "Low",
                "Neutral",
                "Thank you for contacting our customer support team.\n\nYour billing inquiry regarding payment transactions or account charges has been registered.\n\nOur team will review your inquiry and follow up within 24 business hours.\n\nPlease reply directly to this message if you have additional details to add to your case.",
                "Logistic Regression",
                "2026-08-26 13:20:00"
            ),
            (
                "The product exceeded my expectations! Build quality is fantastic and easy to use.",
                "Product Issue",
                "Low",
                "Positive",
                "Thank you for reaching out and for your wonderful feedback!\n\nThank you for reporting this issue with your product. Please review our standard troubleshooting steps and hardware guidelines.\n\nOur team will review your inquiry and follow up within 24 business hours.\n\nPlease reply directly to this message if you have additional details to add to your case.",
                "Logistic Regression",
                "2026-08-26 14:05:00"
            )
        ]
        
        cursor.executemany("""
            INSERT INTO tickets (ticket_text, category, priority, sentiment, suggested_response, model_used, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, seed_samples)
        conn.commit()
        
    conn.close()

def add_ticket(ticket_text: str, category: str, priority: str, sentiment: str, suggested_response: str, model_used: str = "Logistic Regression") -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO tickets (ticket_text, category, priority, sentiment, suggested_response, model_used, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (ticket_text, category, priority, sentiment, suggested_response, model_used, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    ticket_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return ticket_id

def get_all_tickets(limit: int = 100, offset: int = 0, search: str = "", category_filter: str = "", priority_filter: str = "", sentiment_filter: str = ""):
    conn = get_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM tickets WHERE 1=1"
    params = []
    
    if search:
        query += " AND (ticket_text LIKE ? OR suggested_response LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%"])
    if category_filter:
        query += " AND category = ?"
        params.append(category_filter)
    if priority_filter:
        query += " AND priority = ?"
        params.append(priority_filter)
    if sentiment_filter:
        query += " AND sentiment = ?"
        params.append(sentiment_filter)
        
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    tickets = [dict(row) for row in rows]
    conn.close()
    return tickets

def get_ticket_by_id(ticket_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def delete_ticket(ticket_id: int) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tickets WHERE id = ?", (ticket_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted

def get_ticket_stats():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM tickets")
    total_tickets = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM tickets WHERE priority = 'High'")
    high_priority = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM tickets WHERE sentiment = 'Negative'")
    negative_sentiment = cursor.fetchone()[0]
    
    cursor.execute("SELECT category, COUNT(*) as cnt FROM tickets GROUP BY category ORDER BY cnt DESC LIMIT 1")
    top_cat_row = cursor.fetchone()
    top_category = top_cat_row[0] if top_cat_row else "N/A"
    
    # Category Distribution
    cursor.execute("SELECT category, COUNT(*) FROM tickets GROUP BY category")
    category_dist = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Priority Distribution
    cursor.execute("SELECT priority, COUNT(*) FROM tickets GROUP BY priority")
    priority_dist = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Sentiment Distribution
    cursor.execute("SELECT sentiment, COUNT(*) FROM tickets GROUP BY sentiment")
    sentiment_dist = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Recent 5 tickets
    cursor.execute("SELECT * FROM tickets ORDER BY id DESC LIMIT 5")
    recent_tickets = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return {
        "total_tickets": total_tickets,
        "high_priority": high_priority,
        "negative_sentiment": negative_sentiment,
        "top_category": top_category,
        "category_distribution": category_dist,
        "priority_distribution": priority_dist,
        "sentiment_distribution": sentiment_dist,
        "recent_tickets": recent_tickets
    }
