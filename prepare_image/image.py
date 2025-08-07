from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import shutil
import cv2
import numpy as np
import io
from .prepare_image import process_image, restore_image
from doctr.io import DocumentFile
from fastapi import  APIRouter
from starlette.responses import Response

router = APIRouter(prefix="/image", tags=["Image"])

@router.post("/process-image/")
async def process_and_restore(file: UploadFile = File(...)):
    # Lưu file tạm thời
    with open("temp_input.jpg", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Xử lý ảnh
    try:
        processed_img = process_image("temp_input.jpg")
        restored = restore_image(processed_img)

        # Encode ảnh trả về dưới dạng JPEG
        _, img_encoded = cv2.imencode(".jpg", restored)
        return StreamingResponse(
            io.BytesIO(img_encoded.tobytes()),
            media_type="image/jpeg",
            headers={
                "Content-Disposition": 'attachment; filename="restored.jpg"'
            }
        )
    except Exception as e:
        return {"error": str(e)}