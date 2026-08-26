import unittest
import json
from app import app, predict_ticket_attributes
from database.db import init_db, get_ticket_stats, get_all_tickets

class SupportAITestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        with app.app_context():
            init_db()

    def test_dashboard_route(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'SupportAI Dashboard', response.data)

    def test_analyze_page_route(self):
        response = self.client.get('/analyze')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Analyze Support Ticket', response.data)

    def test_history_page_route(self):
        response = self.client.get('/history')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Ticket History Log', response.data)

    def test_performance_page_route(self):
        response = self.client.get('/performance')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Logistic Regression', response.data)
        self.assertIn(b'Random Forest', response.data)

    def test_user_demo_scenario(self):
        """
        Tests the required user demo scenario:
        'My payment was deducted from my account but my order was not placed. I need this resolved urgently.'
        """
        test_input = "My payment was deducted from my account but my order was not placed. I need this resolved urgently."
        res = predict_ticket_attributes(test_input)
        
        print("\n[Test Scenario Output]")
        print(f"Input: {test_input}")
        print(f"Predicted Category: {res['category']} (Confidence: {res['category_confidence']}%)")
        print(f"Predicted Priority: {res['priority']}")
        print(f"Predicted Sentiment: {res['sentiment']} (Confidence: {res['sentiment_confidence']}%)")
        print(f"Suggested Response:\n{res['suggested_response']}")
        
        self.assertEqual(res['category'], "Billing / Payment")
        self.assertEqual(res['priority'], "High")
        self.assertEqual(res['sentiment'], "Negative")
        self.assertTrue(len(res['suggested_response']) > 20)

    def test_api_analyze_endpoint(self):
        payload = {
            "ticket_text": "I cannot log in with my password and need reset instructions."
        }
        response = self.client.post(
            '/api/analyze',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['category'], "Account / Login")
        self.assertIn('ticket_id', data)

    def test_api_stats_endpoint(self):
        response = self.client.get('/api/stats')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('total_tickets', data)
        self.assertIn('category_distribution', data)
        self.assertIn('priority_distribution', data)

    def test_empty_input_validation(self):
        response = self.client.post(
            '/api/analyze',
            data=json.dumps({"ticket_text": ""}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data['success'])

if __name__ == '__main__':
    unittest.main()
