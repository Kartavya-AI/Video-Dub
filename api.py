import os
import shutil
import logging
import tempfile
import asyncio
import uuid
import aiofiles
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import time
from collections import defaultdict, deque

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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

# Enhanced task storage with TTL and cleanup
task_storage: Dict[str, Dict[str, Any]] = {}
task_cleanup_queue = deque()
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "5"))
TASK_TTL = int(os.getenv("TASK_TTL_SECONDS", "3600"))  # 1 hour

# Thread pool for CPU-intensive operations
thread_pool = ThreadPoolExecutor(max_workers=int(os.getenv("MAX_WORKERS", "4")))

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

async def cleanup_expired_tasks():
    """Background task to clean up expired tasks"""
    while True:
        try:
            current_time = time.time()
            expired_tasks = []
            
            for task_id, task_data in task_storage.items():
                if current_time - task_data.get("created_at", 0) > TASK_TTL:
                    expired_tasks.append(task_id)
            
            for task_id in expired_tasks:
                await cleanup_task_files(task_id)
                del task_storage[task_id]
                logger.info("Cleaned up expired task", task_id=task_id)
            
            await asyncio.sleep(300)  # Check every 5 minutes
        except Exception as e:
            logger.error("Error in cleanup task", error=str(e))
            await asyncio.sleep(60)

async def cleanup_task_files(task_id: str):
    """Cleanup files associated with a task"""
    try:
        task_temp_dir = Path("/tmp/video_dubbing") / task_id
        if task_temp_dir.exists():
            shutil.rmtree(task_temp_dir)
        
        # Cleanup output file if exists
        output_path = Path("/tmp/video_dubbing") / f"{task_id}_dubbed.mp4"
        if output_path.exists():
            output_path.unlink()
    except Exception as e:
        logger.warning("Error cleaning up task files", task_id=task_id, error=str(e))

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Video Dubbing API", version="2.0.0")
    
    # Validate environment variables - UPDATED TO INCLUDE OPENAI_API_KEY
    required_env_vars = ["ELEVENLABS_API_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error("Missing required environment variables", missing_vars=missing_vars)
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Create temp directory
    temp_dir = Path("/tmp/video_dubbing")
    temp_dir.mkdir(exist_ok=True, parents=True)
    
    # Start cleanup task
    cleanup_task = asyncio.create_task(cleanup_expired_tasks())
    
    logger.info("Video Dubbing API started successfully")
    yield
    
    # Cleanup on shutdown
    cleanup_task.cancel()
    thread_pool.shutdown(wait=True)
    
    logger.info("Shutting down Video Dubbing API")
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.warning("Error during cleanup", error=str(e))

app = FastAPI(
    title="Video Dubbing API",
    description="AI-powered video dubbing service using CrewAI and ElevenLabs (Optimized)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Optimized middleware configuration
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

async def validate_video_file(file: UploadFile) -> None:
    """Async file validation with streaming checks"""
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Please upload a video file."
        )
    
    # Check file size if available
    max_size = 100 * 1024 * 1024  # 100MB
    if hasattr(file, 'size') and file.size and file.size > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large. Maximum size allowed is 100MB."
        )

async def stream_upload_file(file: UploadFile, destination: Path) -> None:
    """Async streaming file upload to reduce memory usage"""
    try:
        async with aiofiles.open(destination, "wb") as f:
            chunk_size = 1024 * 1024  # 1MB chunks
            while chunk := await file.read(chunk_size):
                await f.write(chunk)
    finally:
        await file.close()

def run_crew_sync(video_path: str, target_language: str) -> str:
    """Synchronous crew execution to run in thread pool"""
    crew = create_crew()
    inputs = {
        'video_path': video_path,
        'target_language': target_language
    }
    
    result = crew.kickoff(inputs=inputs)
    if hasattr(result, 'raw'):
        return result.raw.strip()
    else:
        return str(result).strip()

async def process_dubbing_task(
    task_id: str,
    video_path: Path,
    target_language: str
) -> None:
    """Optimized async dubbing process"""
    try:
        logger.info("Starting dubbing process", task_id=task_id, target_language=target_language)
        
        # Update task status
        task_storage[task_id].update({
            "status": "processing",
            "message": "Processing video for dubbing...",
            "progress": 10
        })
        
        # Run CPU-intensive crew work in thread pool
        logger.info("Running crew for video dubbing", task_id=task_id)
        task_storage[task_id]["progress"] = 30
        
        # Execute crew in thread pool to avoid blocking event loop
        output_path = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            run_crew_sync,
            str(video_path),
            target_language
        )
        
        task_storage[task_id]["progress"] = 80
        
        # Verify output file exists
        if not Path(output_path).exists():
            raise FileNotFoundError(f"Dubbed video file not created: {output_path}")
        
        # Move to final location asynchronously
        final_output_path = Path("/tmp/video_dubbing") / f"{task_id}_dubbed.mp4"
        await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            shutil.move,
            output_path,
            str(final_output_path)
        )

        task_storage[task_id].update({
            "status": "completed",
            "message": "Video dubbing completed successfully",
            "progress": 100,
            "output_path": str(final_output_path),
            "completed_at": time.time()
        })
        
        logger.info("Dubbing process completed successfully", task_id=task_id)
        
    except Exception as e:
        logger.error("Dubbing process failed", task_id=task_id, error=str(e), exc_info=True)
        task_storage[task_id].update({
            "status": "failed",
            "message": f"Dubbing failed: {str(e)}",
            "error_message": str(e),
            "completed_at": time.time()
        })

@app.get("/", summary="Health Check")
async def health_check():
    """Fast health check endpoint"""
    return {
        "status": "ok",
        "service": "Video Dubbing API",
        "version": "2.0.0",
        "timestamp": time.time()
    }

@app.get("/health", summary="Detailed Health Check")
async def detailed_health_check():
    """Detailed health check with system info"""
    env_status = {}
    # UPDATED TO INCLUDE OPENAI_API_KEY IN HEALTH CHECK
    required_vars = ["ELEVENLABS_API_KEY", "OPENAI_API_KEY"]
    
    for var in required_vars:
        env_status[var] = "configured" if os.getenv(var) else "missing"
    
    processing_tasks = len([t for t in task_storage.values() if t["status"] == "processing"])
    
    return {
        "status": "healthy",
        "service": "Video Dubbing API",
        "version": "2.0.0",
        "environment": env_status,
        "active_tasks": processing_tasks,
        "total_tasks": len(task_storage),
        "max_concurrent_tasks": MAX_CONCURRENT_TASKS,
        "available_workers": thread_pool._max_workers - len(thread_pool._threads)
    }

@app.post("/dub-video", response_model=DubbingResponse, summary="Upload and Process Video for Dubbing")
async def start_dubbing_task(
    background_tasks: BackgroundTasks,
    target_language: str = Form("Hindi"),
    video_file: UploadFile = File(...)
):
    """Optimized video upload and dubbing initiation"""
    
    # Check concurrent task limit
    processing_tasks = len([t for t in task_storage.values() if t["status"] == "processing"])
    if processing_tasks >= MAX_CONCURRENT_TASKS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Maximum concurrent tasks ({MAX_CONCURRENT_TASKS}) reached. Please try again later."
        )
    
    # Validate file
    await validate_video_file(video_file)
    
    # Generate task ID and create directory
    task_id = str(uuid.uuid4())
    task_temp_dir = Path("/tmp/video_dubbing") / task_id
    task_temp_dir.mkdir(parents=True, exist_ok=True)
    input_video_path = task_temp_dir / f"input_{video_file.filename}"
    
    try:
        # Stream upload file asynchronously
        await stream_upload_file(video_file, input_video_path)
        
        logger.info(
            "Video uploaded successfully",
            task_id=task_id,
            filename=video_file.filename,
            target_language=target_language,
            file_size=input_video_path.stat().st_size
        )
        
    except Exception as e:
        logger.error("Failed to save uploaded video", task_id=task_id, error=str(e))
        await cleanup_task_files(task_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded video"
        )
    
    # Store task info
    task_storage[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "message": "Task queued for processing",
        "progress": 0,
        "target_language": target_language,
        "filename": video_file.filename,
        "created_at": time.time()
    }
    
    # Start background processing
    background_tasks.add_task(
        process_dubbing_task,
        task_id,
        input_video_path,
        target_language
    )
    
    return DubbingResponse(
        task_id=task_id,
        status="pending",
        message=f"Video dubbing task started. Use /task-status/{task_id} to check progress."
    )

@app.get("/task-status/{task_id}", response_model=TaskStatus, summary="Get Task Status")
async def get_task_status(task_id: str):
    """Fast task status lookup with better error handling"""
    logger.info("Task status request", task_id=task_id)
    
    if task_id not in task_storage:
        # Log available task IDs for debugging
        available_tasks = list(task_storage.keys())
        logger.warning(
            "Task not found", 
            requested_task_id=task_id, 
            available_tasks=available_tasks,
            total_tasks=len(task_storage)
        )
        
        # Provide helpful error message
        if not available_tasks:
            detail = "No tasks found. Please start a dubbing task first using the /dub-video endpoint."
        else:
            detail = f"Task '{task_id}' not found. Available task IDs: {', '.join(available_tasks[:5])}{'...' if len(available_tasks) > 5 else ''}"
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail
        )
    
    task_data = task_storage[task_id]
    logger.info("Task status retrieved", task_id=task_id, status=task_data["status"])
    return TaskStatus(**task_data)

@app.get("/download/{task_id}", summary="Download Dubbed Video")
async def download_dubbed_video(task_id: str):
    """Optimized file download with streaming"""
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
    
    # Use streaming response for large files
    def generate_file():
        with open(output_path, "rb") as f:
            while chunk := f.read(8192):  # 8KB chunks
                yield chunk
    
    file_size = output_path.stat().st_size
    filename = f"dubbed_{task_data['filename']}"
    
    return StreamingResponse(
        generate_file(),
        media_type='video/mp4',
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Length": str(file_size)
        }
    )

@app.delete("/task/{task_id}", summary="Cancel or Delete Task")
async def delete_task(task_id: str):
    """Async task deletion with proper cleanup"""
    if task_id not in task_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    # Cleanup files asynchronously
    await cleanup_task_files(task_id)
    
    # Remove from storage
    del task_storage[task_id]
    logger.info("Task deleted successfully", task_id=task_id)
    
    return {"message": "Task deleted successfully", "task_id": task_id}

@app.get("/tasks", summary="List All Tasks")
async def list_tasks(
    status_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """Paginated task listing"""
    tasks = list(task_storage.values())
    
    # Apply status filter
    if status_filter:
        tasks = [t for t in tasks if t["status"] == status_filter]
    
    # Apply pagination
    total = len(tasks)
    tasks = tasks[offset:offset + limit]
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "tasks": tasks
    }

# NEW: Create a demo task endpoint for testing
@app.post("/create-demo-task", summary="Create Demo Task for Testing")
async def create_demo_task(task_id: Optional[str] = None):
    """Create a demo task for testing purposes"""
    if not task_id:
        task_id = str(uuid.uuid4())
    
    # Check if task already exists
    if task_id in task_storage:
        return {
            "message": f"Demo task {task_id} already exists",
            "task_id": task_id,
            "status": task_storage[task_id]["status"]
        }
    
    # Create demo task
    task_storage[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "message": "Demo task created for testing",
        "progress": 0,
        "target_language": "Hindi",
        "filename": "demo_video.mp4",
        "created_at": time.time()
    }
    
    logger.info("Demo task created", task_id=task_id)
    
    return {
        "message": f"Demo task created successfully",
        "task_id": task_id,
        "status": "pending"
    }

@app.get("/metrics", summary="System Metrics")
async def get_metrics():
    """System performance metrics"""
    return {
        "active_tasks": len([t for t in task_storage.values() if t["status"] == "processing"]),
        "total_tasks": len(task_storage),
        "max_concurrent_tasks": MAX_CONCURRENT_TASKS,
        "thread_pool_size": thread_pool._max_workers,
        "active_threads": len(thread_pool._threads),
        "task_ttl_seconds": TASK_TTL,
        "uptime": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        workers=1,  # Use 1 worker with async/threading
        reload=False,
        log_level="info",
        access_log=False,  # Disable access logs for performance
        loop="asyncio",
        http="httptools",  # Faster HTTP parser
        lifespan="on"
    )