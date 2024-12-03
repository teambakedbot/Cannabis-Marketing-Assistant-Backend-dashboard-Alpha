# CANNABIS-MARKETING-ASSISTANT

## Overview

This project is a web application built with **FastAPI**, designed to manage and interact with various resources such as products, users, and dispensaries. It leverages Redis for caching and data storage, and integrates with external services for email and SMS notifications.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
  - [1. Create a Virtual Environment](#1-create-a-virtual-environment)
  - [2. Activate the Virtual Environment](#2-activate-the-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Configure Environment Variables](#4-configure-environment-variables)
  - [5. Start Redis Server](#5-start-redis-server)
  - [6. Run the Application](#6-run-the-application)
  - [7. Run Data Ingestion](#7-run-data-ingestion)
  - [8. Run Dataset Migration](#8-run-dataset-migration)
- [Testing](#testing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Prerequisites

Before running the project, ensure you have the following installed:

- **Python 3.11** or higher
- **Redis server**

## Setup Instructions

### 1. Create a Virtual Environment

To create a virtual environment, run the following command:

```bash
python3 -m venv myenv
```

### 2. Activate the Virtual Environment

Activate your virtual environment using the command appropriate for your operating system:

```bash
# On macOS/Linux
source myenv/bin/activate

# On Windows
myenv\Scripts\activate
```

### 3. Install Dependencies

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory of the project and add the necessary environment variables. Here’s an example of what to include:

```env
# .env file
REDIS_URL=redis://localhost:6379/0
CANNMENUS_API_KEY=your_api_key_here
SENDGRID_API_KEY=your_sendgrid_api_key_here
TWILIO_ACCOUNT_SID=your_twilio_account_sid_here
TWILIO_AUTH_TOKEN=your_twilio_auth_token_here
TWILIO_PHONE_NUMBER=your_twilio_phone_number_here
```

### 5. Start Redis Server

Make sure your Redis server is running. You can start it using the following command:

```bash
redis-server
```

### 6. Run the Application

To run the FastAPI application, use the following command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 7. Run Data Ingestion

If you need to run data ingestion, execute:

```bash
python app/data_ingestion.py
```

### 8. Run Dataset Migration

To migrate datasets, run:

```bash
python app/migrate_datasets.py
```

## Testing

Ensure that your Redis server is running correctly before starting the application. You can test the connection by using a Redis client or command line.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **FastAPI** for the web framework
- **Redis** for caching and data storage
- **SendGrid** for email services
- **Twilio** for SMS services
