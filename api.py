import os
import shutil
import logging
import tempfile
import asyncio
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
import structlog
from src.video_dubbing.crew import create_crew

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

task_storage: Dict[str, Dict[str, Any]] = {}

class DubbingRequest(BaseModel):
    target_language: str = Field(default="Hindi", description="Target language for dubbing")
    
class DubbingResponse(BaseModel):
    task_id: str = Field(description="Unique task identifier")
    status: str = Field(description="Task status: pending, processing, completed, failed")
    message: str = Field(description="Status message")
    video_url: Optional[str] = Field(None, description="Download URL for completed video")
    
class TaskStatus(BaseModel):
    task_id: str
    status: str
    message: str
    progress: Optional[int] = Field(None, description="Progress percentage (0-100)")
    created_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Video Dubbing API", version="1.0.0")
    required_env_vars = ["ELEVENLABS_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error("Missing required environment variables", missing_vars=missing_vars)
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")
    

    temp_dir = Path("/tmp/video_dubbing")
    temp_dir.mkdir(exist_ok=True, parents=True)
    logger.info("Video Dubbing API started successfully")
    yield
    logger.info("Shutting down Video Dubbing API")
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.warning("Error during cleanup", error=str(e))

app = FastAPI(
    title="Video Dubbing API",
    description="AI-powered video dubbing service using CrewAI and ElevenLabs",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(
        "Unexpected exception occurred",
        error=str(exc),
        path=request.url.path,
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

def validate_video_file(file: UploadFile) -> None:
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Please upload a video file."
        )
    max_size = 100 * 1024 * 1024
    if hasattr(file, 'size') and file.size and file.size > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large. Maximum size allowed is 100MB."
        )

async def process_dubbing_task(
    task_id: str,
    video_path: Path,
    target_language: str
) -> None:
    try:
        logger.info("Starting dubbing process", task_id=task_id, target_language=target_language)
        task_storage[task_id].update({
            "status": "processing",
            "message": "Processing video for dubbing...",
            "progress": 10
        })
        crew = create_crew()
        inputs = {
            'video_path': str(video_path),
            'target_language': target_language
        }
        
        logger.info("Running crew for video dubbing", task_id=task_id)
        task_storage[task_id]["progress"] = 50
        
        result = crew.kickoff(inputs=inputs)
        if hasattr(result, 'raw'):
            output_path = result.raw.strip()
        else:
            output_path = str(result).strip()
        if not Path(output_path).exists():
            raise FileNotFoundError(f"Dubbed video file not created: {output_path}")
        final_output_path = Path("/tmp/video_dubbing") / f"{task_id}_dubbed.mp4"
        shutil.move(output_path, final_output_path)

        task_storage[task_id].update({
            "status": "completed",
            "message": "Video dubbing completed successfully",
            "progress": 100,
            "output_path": str(final_output_path),
            "completed_at": str(asyncio.get_event_loop().time())
        })
        
        logger.info("Dubbing process completed successfully", task_id=task_id)
        
    except Exception as e:
        logger.error("Dubbing process failed", task_id=task_id, error=str(e), exc_info=True)
        task_storage[task_id].update({
            "status": "failed",
            "message": f"Dubbing failed: {str(e)}",
            "error_message": str(e),
            "completed_at": str(asyncio.get_event_loop().time())
        })

@app.get("/", summary="Health Check")
async def health_check():
    return {
        "status": "ok",
        "service": "Video Dubbing API",
        "version": "1.0.0",
        "timestamp": asyncio.get_event_loop().time()
    }

@app.get("/health", summary="Detailed Health Check")
async def detailed_health_check():
    env_status = {}
    required_vars = ["ELEVENLABS_API_KEY"]
    
    for var in required_vars:
        env_status[var] = "configured" if os.getenv(var) else "missing"
    return {
        "status": "healthy",
        "service": "Video Dubbing API",
        "version": "1.0.0",
        "environment": env_status,
        "active_tasks": len([t for t in task_storage.values() if t["status"] == "processing"]),
        "total_tasks": len(task_storage)
    }

@app.post("/dub-video", response_model=DubbingResponse, summary="Upload and Process Video for Dubbing")
async def start_dubbing_task(
    background_tasks: BackgroundTasks,
    target_language: str = Form("Hindi"),
    video_file: UploadFile = File(...)
):

    validate_video_file(video_file)
    task_id = str(uuid.uuid4())
    task_temp_dir = Path("/tmp/video_dubbing") / task_id
    task_temp_dir.mkdir(parents=True, exist_ok=True)
    input_video_path = task_temp_dir / f"input_{video_file.filename}"
    try:
        with open(input_video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        
        logger.info(
            "Video uploaded successfully",
            task_id=task_id,
            filename=video_file.filename,
            target_language=target_language
        )
        
    except Exception as e:
        logger.error("Failed to save uploaded video", task_id=task_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded video"
        )
    finally:
        video_file.file.close()
    task_storage[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "message": "Task queued for processing",
        "progress": 0,
        "target_language": target_language,
        "filename": video_file.filename,
        "created_at": str(asyncio.get_event_loop().time())
    }
    
    background_tasks.add_task(
        process_dubbing_task,
        task_id,
        input_video_path,
        target_language
    )
    
    return DubbingResponse(
        task_id=task_id,
        status="pending",
        message="Video dubbing task started. Use /task-status/{task_id} to check progress."
    )

@app.get("/task-status/{task_id}", response_model=TaskStatus, summary="Get Task Status")
async def get_task_status(task_id: str):
    if task_id not in task_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    task_data = task_storage[task_id]
    return TaskStatus(**task_data)

@app.get("/download/{task_id}", summary="Download Dubbed Video")
async def download_dubbed_video(task_id: str):
    if task_id not in task_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    task_data = task_storage[task_id]
    
    if task_data["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task is not completed. Current status: {task_data['status']}"
        )
    
    output_path = Path(task_data.get("output_path", ""))
    
    if not output_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dubbed video file not found"
        )
    
    return FileResponse(
        path=str(output_path),
        media_type='video/mp4',
        filename=f"dubbed_{task_data['filename']}",
        headers={"Content-Disposition": f"attachment; filename=dubbed_{task_data['filename']}"}
    )

@app.delete("/task/{task_id}", summary="Cancel or Delete Task")
async def delete_task(task_id: str):
    if task_id not in task_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    task_data = task_storage[task_id]
    task_temp_dir = Path("/tmp/video_dubbing") / task_id
    if task_temp_dir.exists():
        shutil.rmtree(task_temp_dir)
    
    if "output_path" in task_data:
        output_path = Path(task_data["output_path"])
        if output_path.exists():
            output_path.unlink()
    
    del task_storage[task_id]
    logger.info("Task deleted successfully", task_id=task_id)
    return {"message": "Task deleted successfully", "task_id": task_id}

@app.get("/tasks", summary="List All Tasks")
async def list_tasks(status_filter: Optional[str] = None):
    tasks = list(task_storage.values())
    
    if status_filter:
        tasks = [t for t in tasks if t["status"] == status_filter]
    return {
        "total": len(tasks),
        "tasks": tasks
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        log_level="info"
    )