import os
import shutil
import uuid
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel, HttpUrl

try:
    from src.api.sync_api import SyncAPI
except ImportError:
    class MockSyncAPI:
        def __init__(self, temp_dir: str):
            pass

        async def process_single_video(self, video_path: str) -> Dict[str, Any]:
            if "fail" in video_path.lower():
                raise ValueError("Simulated processing failure for this video.")

            return {
                "input_source": video_path,
                "extracted_cues": [
                    {"timestamp_ms": 1000, "type": "pitch_shift", "value": 5.2, "confidence": 0.95},
                    {"timestamp_ms": 2500, "type": "loudness_peak", "value": -15.0, "confidence": 0.88}
                ],
                "audio_quality_metadata": {"snr_db": 25.5, "reverb_time_s": 0.3},
                "processing_status": "success",
                "message": "Mock processing completed successfully."
            }

        async def process_video_batch(self, video_sources: List[str]) -> List[Dict[str, Any]]:
            results = []
            for source in video_sources:
                try:
                    result = await self.process_single_video(source)
                    results.append(result)
                except ValueError as e:
                    results.append({
                        "input_source": source,
                        "processing_status": "failed",
                        "message": str(e)
                    })
                except Exception as e:
                    results.append({
                        "input_source": source,
                        "processing_status": "failed",
                        "message": f"An unexpected error occurred: {e}"
                    })
            return results

    SyncAPI = MockSyncAPI

app = FastAPI(
    title="Robust Audio-Visual Synchronization Engine API",
    description="A microservice for extracting reliable, confidence-scored temporal cues from 'in-the-wild' videos.",
    version="1.0.0"
)

TEMP_UPLOAD_DIR = "/tmp/sync_web_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
sync_engine = SyncAPI(temp_dir=TEMP_UPLOAD_DIR)


class ProcessVideoResponse(BaseModel):
    input_source: str
    extracted_cues: List[Dict[str, Any]]
    audio_quality_metadata: Dict[str, Any]
    processing_status: str
    message: str


class BatchProcessRequest(BaseModel):
    video_urls: List[HttpUrl]


class BatchProcessResponse(BaseModel):
    results: List[ProcessVideoResponse]
    overall_status: str


@app.get("/health", summary="Health check endpoint")
async def health_check():
    return {"status": "healthy", "message": "Robust Audio-Visual Synchronization Engine is running."}


@app.post(
    "/process/video",
    response_model=ProcessVideoResponse,
    summary="Process a single video file for temporal cues",
    status_code=status.HTTP_200_OK
)
async def process_single_video_endpoint(
    video_file: UploadFile = File(..., description="The video file to process.")
):
    if not video_file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded.")

    temp_file_path = None
    try:
        unique_filename = f"{uuid.uuid4()}_{video_file.filename}"
        temp_file_path = os.path.join(TEMP_UPLOAD_DIR, unique_filename)

        with open(temp_file_path, "wb") as buffer:
            while True:
                chunk = await video_file.read(8192)
                if not chunk:
                    break

        result = await sync_engine.process_single_video(temp_file_path)
        
        return ProcessVideoResponse(**result)

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process video '{video_file.filename}': {e}"
        )
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                print(f"WARNING: Failed to clean up temporary file {temp_file_path}: {e}")


@app.post(
    "/process/batch-urls",
    response_model=BatchProcessResponse,
    summary="Process multiple videos via URLs for temporal cues",
    status_code=status.HTTP_200_OK
)
async def process_batch_urls_endpoint(
    request: BatchProcessRequest
):
    if not request.video_urls:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No video URLs provided.")

    video_sources = [str(url) for url in request.video_urls]
    
    try:
        raw_results = await sync_engine.process_video_batch(video_sources)
        
        processed_results = [ProcessVideoResponse(**r) for r in raw_results]
        
        overall_status = "success"
        if any(r.processing_status == "failed" for r in processed_results):
            if all(r.processing_status == "failed" for r in processed_results):
                overall_status = "all_failed"
            else:
                overall_status = "partial_failure"

        return BatchProcessResponse(results=processed_results, overall_status=overall_status)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process batch: {e}"
        )


@app.on_event("shutdown")
async def shutdown_event():
    if os.path.exists(TEMP_UPLOAD_DIR):
        try:
            shutil.rmtree(TEMP_UPLOAD_DIR)
        except Exception as e:
            print(f"WARNING: Failed to clean up temporary directory {TEMP_UPLOAD_DIR} on shutdown: {e}")