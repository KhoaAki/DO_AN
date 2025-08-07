from pytesseract import Output
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"main_function/OCR/Tesseract-OCR/tesseract.exe"
from fastapi import  APIRouter
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import PlainTextResponse
import shutil
import cv2
import numpy as np
import pytesseract
import tempfile
import os
import language_tool_python
# --- HuggingFace Spell Correction cho tiếng Việt ---
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel

router = APIRouter(prefix="/image", tags=["Image"])

vi_tokenizer = AutoTokenizer.from_pretrained("bmd1905/vietnamese-correction-v2")
vi_model = AutoModelForSeq2SeqLM.from_pretrained("bmd1905/vietnamese-correction-v2")
def correct_vietnamese_text(text: str) -> str:
    inputs = vi_tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = vi_model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    corrected_text = vi_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

# --- Hàm OCR với Tesseract ---
def ocr_with_tesseract_only(image: np.ndarray, lang='eng') -> str:
    if image is None or image.size == 0:
        raise ValueError("❌ Ảnh không hợp lệ hoặc rỗng.")
    if len(image.shape) == 2 or image.shape[2] == 1:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image
    text = pytesseract.image_to_string(image_rgb, lang=lang, config='--psm 6').strip()
    return text

# --- Endpoint xử lý ảnh ---
@router.post("/ocr-image/", response_class=PlainTextResponse)
async def ocr_image_tesseract(file: UploadFile = File(...), lang: str = Form(default='eng')):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)
        image = cv2.imread(tmp_path)
        raw_text = ocr_with_tesseract_only(image, lang)
        os.remove(tmp_path)

        if not raw_text:
            return "⚠ Không phát hiện được văn bản nào."

        # Nếu là tiếng Anh: sửa bằng LanguageTool
        if lang.startswith("en"):
            import language_tool_python
            import re
            tool = language_tool_python.LanguageTool('en-US')
            corrected_lines = []
            for line in raw_text.splitlines():
                if not line.strip():
                    corrected_lines.append("")
                    continue
                sentences = re.split(r'(?<=[.?!])\s+', line)
                corrected_sentences = [tool.correct(s.strip()) for s in sentences if s.strip()]
                corrected_line = ' '.join(corrected_sentences)
                corrected_lines.append(corrected_line)
            corrected_text = '\n'.join(corrected_lines)
            return corrected_text

        # Nếu là tiếng Việt: sửa lỗi bằng mô hình huggingface
        elif lang.startswith("vie"):
            corrected = correct_vietnamese_text(raw_text)
            return corrected

        # Ngôn ngữ khác: trả về raw
        return raw_text

    except Exception as e:
        return f"❌ Lỗi xử lý ảnh: {str(e)}"