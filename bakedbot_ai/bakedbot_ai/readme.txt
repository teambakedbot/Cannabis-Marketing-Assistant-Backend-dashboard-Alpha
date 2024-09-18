# BakedBot AI

BakedBot AI is a comprehensive AI-driven platform designed for the cannabis industry, consisting of Craig (AI marketing assistant) and Smokey (AI-powered chatbot and product locator).

## Features

- Craig: AI-powered marketing assistant for dispensaries
- Smokey: Intelligent chatbot for product recommendations and dispensary locations
- Dynamic product and strain database
- Personalized user interactions
- Advanced AI and machine learning implementation
- Robust security and compliance checks

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/bakedbot-ai.git
   cd bakedbot-ai
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and fill in your environment variables:
   ```
   cp .env.example .env
   ```

5. Run database migrations:
   ```
   alembic upgrade head
   ```

6. Start the FastAPI server:
   ```
   uvicorn main:app --reload
   ```

## Running Tests

To run tests, use the following command:
```
pytest
```

## API Documentation

API documentation is available at `/docs` when the server is running.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
