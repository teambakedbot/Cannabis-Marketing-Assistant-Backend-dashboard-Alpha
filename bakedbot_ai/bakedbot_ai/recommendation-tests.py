import unittest
from unittest.mock import patch, MagicMock
from .recommendation_system import get_recommendations
from .models import Product, User, Interaction

class TestRecommendationSystem(unittest.TestCase):

    @patch('recommendation_system.SessionLocal')
    @patch('recommendation_system.pinecone.Index')
    def test_get_recommendations(self, mock_index, mock_session):
        # Mock database session
        mock_db = MagicMock()
        mock_session.return_value = mock_db

        # Mock user and interactions
        mock_user = User(id=1)
        mock_interactions = [
            Interaction(user_id=1, product_id=1, rating=5),
            Interaction(user_id=1, product_id=2, rating=4),
        ]
        mock_db.query().filter().first.return_value = mock_user
        mock_db.query().filter().all.return_value = mock_interactions

        # Mock products
        mock_products = [
            Product(id=1, name="Product 1"),
            Product(id=2, name="Product 2"),
            Product(id=3, name="Product 3"),
        ]
        mock_db.query().filter().first.side_effect = mock_products

        # Mock Pinecone results
        mock_index.return_value.query.return_value.matches = [
            MagicMock(id="1"), MagicMock(id="2"), MagicMock(id="3")
        ]

        # Call the function
        recommendations = get_recommendations(1, n=3)

        # Assertions
        self.assertEqual(len(recommendations), 3)
        self.assertEqual(recommendations[0].name, "Product 1")
        self.assertEqual(recommendations[1].name, "Product 2")
        self.assertEqual(recommendations[2].name, "Product 3")

if __name__ == '__main__':
    unittest.main()

