# Video Dubbing Agent
## Overview

This project provides an AI-powered video dubbing solution that converts English videos to Hindi using advanced AI tools including:
- **Whisper AI** for audio transcription
- **OpenAI GPT** for translation
- **ElevenLabs API** for high-quality voice synthesis
- **CrewAI** for orchestrating the multi-agent workflow

The system supports both web API and Streamlit interfaces for easy integration and user interaction.

## Installation

Ensure you have Python >=3.10 <3.14 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
```

### Manual Installation

Alternatively, you can install dependencies using pip:

```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory and add your API keys:

```bash
OPENAI_API_KEY=your_openai_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

### Customizing

- Modify `src/video_dubbing/config/agents.yaml` to define your agents
- Modify `src/video_dubbing/config/tasks.yaml` to define your tasks
- Modify `src/video_dubbing/crew.py` to add your own logic, tools and specific args
- Modify `src/video_dubbing/main.py` to add custom inputs for your agents and tasks

## Running the Project

### Local Development

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
crewai run
```

### Web Interface (Streamlit)

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

### API Server

Start the FastAPI server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

## Docker Deployment

### Building the Docker Image

The project includes a comprehensive Dockerfile for containerized deployment:

```bash
docker build -t video-dubbing-app .
```

### Running with Docker

```bash
docker run -p 8080:8080 -e OPENAI_API_KEY=your_key -e ELEVENLABS_API_KEY=your_key video-dubbing-app
```

### Docker Configuration

The Dockerfile uses:
- **Base Image**: `python:3.11-slim-bullseye`
- **Port**: 8080 (configurable)
- **Server**: Gunicorn with Uvicorn workers
- **Configuration**: 1 worker, 2 threads for optimal performance

Key Docker features:
- Optimized Python environment with no cache writes
- Proper PYTHONPATH configuration
- Production-ready Gunicorn server setup
- Efficient layer caching with requirements.txt copied first

## Google Cloud Platform (GCP) Deployment

### Prerequisites

1. Install and configure Google Cloud SDK
2. Ensure you have a GCP project with billing enabled
3. Enable required APIs

### Step-by-Step GCP Deployment

#### 1. Authentication and Project Setup

```bash
# Login to Google Cloud
gcloud auth login

# List available projects
gcloud projects list

# Set your project ID
gcloud config set project YOUR_PROJECT_ID

# Enable required services
gcloud services enable cloudbuild.googleapis.com artifactregistry.googleapis.com run.googleapis.com
```

#### 2. Create Artifact Registry Repository

```bash
# Set variables (customize these values)
$REPO_NAME = "video-dubbing-repo"
$REGION = "us-central1"

# Create Docker repository
gcloud artifacts repositories create $REPO_NAME `
    --repository-format=docker `
    --location=$REGION `
    --description="Video dubbing application repository"
```

#### 3. Build and Push Docker Image

```bash
# Get project ID
$PROJECT_ID = $(gcloud config get-value project)

# Create image tag
$IMAGE_TAG = "$($REGION)-docker.pkg.dev/$($PROJECT_ID)/$($REPO_NAME)/video-dubbing:latest"

# Build and push image to Artifact Registry
gcloud builds submit --tag $IMAGE_TAG
```

#### 4. Deploy to Cloud Run

```bash
# Set service name
$SERVICE_NAME = "video-dubbing-service"

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME `
    --image=$IMAGE_TAG `
    --platform=managed `
    --region=$REGION `
    --allow-unauthenticated `
    --set-env-vars="OPENAI_API_KEY=your_openai_key,ELEVENLABS_API_KEY=your_elevenlabs_key" `
    --memory=2Gi `
    --cpu=2 `
    --timeout=3600 `
    --max-instances=10
```

### Environment Variables for Production

When deploying to GCP, ensure you set the following environment variables:

```bash
OPENAI_API_KEY=your_openai_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

### GCP Deployment Features

- **Serverless**: Automatic scaling with Cloud Run
- **Container Registry**: Secure image storage with Artifact Registry
- **Build Automation**: Cloud Build for CI/CD
- **High Availability**: Multi-region deployment support
- **Cost Optimization**: Pay-per-use pricing model

### Monitoring and Logging

Access logs and monitoring through:
- **Cloud Logging**: `gcloud logging read "resource.type=cloud_run_revision"`
- **Cloud Monitoring**: Built-in metrics for Cloud Run services
- **Error Reporting**: Automatic error detection and reporting

## API Endpoints

### FastAPI Endpoints

- **GET /** - Health check endpoint
- **POST /dub-video/** - Upload video and get dubbed version
  - Parameters:
    - `target_language`: Target language (default: "Hindi")
    - `video_file`: Video file to dub (multipart/form-data)

### Usage Example

```bash
curl -X POST "http://your-service-url/dub-video/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "target_language=Hindi" \
  -F "video_file=@your_video.mp4"
```

## Understanding Your Crew

The video_dubbing Crew is composed of a specialized AI agent:

**Video Dubber Agent**: Expert in media translation and dubbing, specializing in converting English video content to Hindi while maintaining original timing, context, and emotional tone.

The agent uses the ElevenLabs Dubbing API for professional-quality voice synthesis and video processing.

## File Structure

```
video_dubbing/
├── src/video_dubbing/
│   ├── config/
│   │   ├── agents.yaml          # Agent configurations
│   │   └── tasks.yaml           # Task definitions
│   ├── tools/
│   │   └── custom_tool.py       # ElevenLabs dubbing tool
│   ├── crew.py                  # Crew orchestration
│   └── main.py                  # Main execution logic
├── api.py                       # FastAPI application
├── app.py                       # Streamlit web interface
├── Dockerfile                   # Container configuration
├── requirements.txt             # Python dependencies
├── .dockerignore               # Docker ignore patterns
└── README.md                   # This file
```

## Troubleshooting

### Common Issues

1. **API Key Issues**: Ensure all required API keys are properly set in environment variables
2. **File Upload Limits**: Check file size limits for your deployment platform
3. **Memory Limits**: Video processing requires adequate memory allocation
4. **Network Timeouts**: Large files may require increased timeout settings

### GCP Specific Issues

1. **Permissions**: Ensure proper IAM roles are assigned
2. **Quotas**: Check GCP quotas for Cloud Run and Artifact Registry
3. **Region Availability**: Verify services are available in your chosen region

## Support

For support, questions, or feedback regarding the VideoDubbing Crew or crewAI:
- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)

Let's create wonders together with the power and simplicity of crewAI.
