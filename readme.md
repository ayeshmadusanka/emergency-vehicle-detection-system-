# Emergency Vehicle Detection System

This project is a smart traffic management system that uses AI to detect emergency vehicles and adjust traffic signals accordingly.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:ayeshmadusanka/emergency-vehicle-detection-system-.git
    cd emergency-vehicle-detection-system-
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Environment Variables

Create a `.env` file in the root of the project and add the following environment variables.

### For Gemini AI Detector:
```
GOOGLE_AI_API_KEY=your_google_ai_api_key
```

### For Vertex AI Detector:
```
GAPI_PROJECT_ID=your_gcp_project_id
GAPI_CLIENT_EMAIL=your_service_account_email
GAPI_PRIVATE_KEY=your_service_account_private_key
GAPI_TOKEN_URI=https://oauth2.googleapis.com/token
TYPE=service_account
PRIVATE_KEY_ID=your_private_key_id
```

## How to Run

There are two applications in this project:

1.  **Main Application (`app.py`):** A smart traffic management system.
    ```bash
    python app.py
    ```
    The application will be available at `http://localhost:8003`.

2.  **Vertex AI Application (`vertex_app.py`):** An emergency vehicle detection system using Vertex AI.
    ```bash
    python vertex_app.py
    ```
    The application will be available at `http://localhost:8005`.

## Usage

-   **Emergency Detection:** Access the main interface at `http://localhost:8003/` to upload an image and get a prediction.
-   **Traffic Management:** Access the traffic management interface at `http://localhost:8003/traffic`.
-   **Health Check:** Check the health of the application at `http://localhost:8003/health`.
