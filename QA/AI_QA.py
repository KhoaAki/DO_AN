from fastapi import FastAPI, UploadFile, File, Form,APIRouter
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import io
from io import BytesIO

router = APIRouter(prefix="/image", tags=["Image"])

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
device = "cuda" if torch.cuda.is_available() else "cpu"

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1)
         for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num},
        key=lambda x: x[0] * x[1]
    )
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_bytes, input_size=448, max_num=12):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


# Load model and tokenizer
model_name = "5CD-AI/Vintern-1B-v3_5"
try:
  model = AutoModel.from_pretrained(
      model_name,
      torch_dtype=torch.bfloat16,
      low_cpu_mem_usage=True,
      trust_remote_code=True,
      use_flash_attn=False,
  ).eval().to(device)
except:
  model = AutoModel.from_pretrained(
      model_name,
      torch_dtype=torch.bfloat16,
      low_cpu_mem_usage=True,
      trust_remote_code=True
  ).eval().to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

generation_config = dict(max_new_tokens=512, do_sample=False, num_beams=3, repetition_penalty=3.5)
@router.post("/qa")
async def image_qa(file: UploadFile = File(...), question: str = Form(...)):
    try:
        content = await file.read()
        pixel_values = load_image(content, max_num=3).to(torch.bfloat16).to(device)
        # Kiểm tra token length của câu hỏi
        tokens = tokenizer(question, return_tensors="pt")
        max_length = getattr(model.config, "max_position_embeddings", 1700)
        if tokens["input_ids"].shape[1] > max_length:
            return JSONResponse(
                status_code=400,
                content={"error": f"Câu hỏi quá dài ({tokens['input_ids'].shape[1]} tokens), vượt quá giới hạn {max_length} tokens."}
            )
        answer = model.chat(tokenizer, pixel_values, question, generation_config)
        return JSONResponse(content={"question": question, "answer": answer})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})