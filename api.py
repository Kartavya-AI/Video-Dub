import os
import shutil
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from src.video_dubbing.crew import create_crew

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Video Dubbing API",
    description="Upload a video, wait for processing, and get the dubbed version back.",
    version="1.0.0"
)

@app.get("/", summary="Health Check")
def read_root():
    return {"status": "ok"}

@app.post("/dub-video/", summary="Upload and Dub Video")
def dub_video_and_download(
    target_language: str = Form("Hindi"),
    video_file: UploadFile = File(...)
):
    if not video_file.content_type or not video_file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        input_video_path = temp_dir_path / video_file.filename
        
        try:
            with open(input_video_path, "wb") as buffer:
                shutil.copyfileobj(video_file.file, buffer)
            logging.info(f"Saved temporary video to {input_video_path}")
        finally:
            video_file.file.close()

        try:
            logging.info("Kicking off the dubbing crew...")
            inputs = {
                'video_path': str(input_video_path),
                'target_language': target_language
            }
            crew = create_crew().crew()
            crew.kickoff(inputs=inputs)
            logging.info("Crew finished processing.")
            output_video_path = Path("dubbed_video.mp4")

            if not output_video_path.exists():
                logging.error("Dubbing finished, but the output file 'dubbed_video.mp4' was not found.")
                raise HTTPException(status_code=500, detail="Processing failed to create the dubbed video.")
                
            final_output_path = temp_dir_path / "final_dubbed_video.mp4"
            shutil.move(str(output_video_path), str(final_output_path))
            return FileResponse(
                path=str(final_output_path),
                media_type='video/mp4',
                filename="dubbed_video.mp4"
            )

        except Exception as e:
            logging.error(f"An error occurred during the dubbing process: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")