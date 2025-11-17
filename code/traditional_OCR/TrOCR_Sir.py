# ======================================================
#  TrOCR Handwritten OCR Pipeline for Scanned PDF Scripts
# ======================================================
# Author: OpenAI Assistant (GPT-5)
# Description:
# - Convert PDF pages to images
# - Preprocess (grayscale, threshold, denoise)
# - Segment handwritten lines
# - Run TrOCR model for text recognition
# - Compare output with ground truth from DOCX
# ======================================================

# --- STEP 0: Install dependencies (uncomment if needed) ---
# !pip install opencv-python pillow numpy transformers torch torchvision jiwer matplotlib pdf2image python-docx

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from jiwer import wer, cer
from pdf2image import convert_from_path
import docx
import torch


# ------------------------------------------------------
# STEP 1: Preprocess scanned page
# ------------------------------------------------------
def preprocess_image(pil_image):
    """Preprocess scanned handwritten page: grayscale, normalize illumination, threshold."""
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Normalize illumination using morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    norm = cv2.divide(gray, bg, scale=255)

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 15)

    # Denoise small specks
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return binary


# ------------------------------------------------------
# STEP 2: Line segmentation
# ------------------------------------------------------
def segment_lines(binary_img, min_height=20):
    """Segment text lines from binary image."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 3))
    dilated = cv2.dilate(binary_img, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for cnt in sorted(contours, key=lambda x: cv2.boundingRect(x)[1]):
        x, y, w, h = cv2.boundingRect(cnt)
        if h > min_height:
            line = binary_img[y:y + h, x:x + w]
            lines.append(line)
    return lines


# ------------------------------------------------------
# STEP 3: Visualize segmented lines (optional)
# ------------------------------------------------------
def visualize_lines(lines, max_lines=5):
    plt.figure(figsize=(10, 10))
    for i, l in enumerate(lines[:max_lines]):
        plt.subplot(max_lines, 1, i + 1)
        plt.imshow(255 - l, cmap='gray')
        plt.axis('off')
    plt.show()


# ------------------------------------------------------
# STEP 4: Initialize TrOCR model
# ------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)


# ------------------------------------------------------
# STEP 5: Recognize a single line
# ------------------------------------------------------
def recognize_line(line_img):
    """Recognize handwritten text line using TrOCR."""
    image = Image.fromarray(255 - line_img).convert("RGB")  # invert for TrOCR
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


# ------------------------------------------------------
# STEP 6: Process a full PDF
# ------------------------------------------------------
def recognize_pdf(pdf_path, dpi=300):
    """Convert PDF to images, preprocess, segment, recognize each page."""
    pages = convert_from_path(pdf_path, dpi=dpi)
    all_page_texts = []

    for i, page in enumerate(pages, 1):
        print(f"\n📄 Processing Page {i}/{len(pages)} ...")
        binary = preprocess_image(page)
        lines = segment_lines(binary)

        page_texts = []
        for idx, line in enumerate(lines):
            txt = recognize_line(line)
            page_texts.append(txt)
            print(f"  Line {idx + 1}: {txt}")

        all_page_texts.append("\n".join(page_texts))
    return all_page_texts


# ------------------------------------------------------
# STEP 7: Read ground truth from DOCX
# ------------------------------------------------------
def read_docx_text(docx_path):
    doc = docx.Document(docx_path)
    text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    return text


# ------------------------------------------------------
# STEP 8: Evaluate OCR output
# ------------------------------------------------------
def evaluate_text(predicted_text, ground_truth_text):
    cer_score = cer(ground_truth_text.lower(), predicted_text.lower())
    wer_score = wer(ground_truth_text.lower(), predicted_text.lower())
    print("\n==================== OCR EVALUATION ====================")
    print(f"CER: {cer_score:.4f}")
    print(f"WER: {wer_score:.4f}")
    print("========================================================")
    return cer_score, wer_score


# ------------------------------------------------------
# STEP 9: Main execution
# ------------------------------------------------------
pdf_path = "a01-007u.pdf"  # input PDF of handwritten answer script
gt_path = "a01-007u.docx"  # ground truth DOCX file

print("🚀 Starting OCR Pipeline")
page_texts = recognize_pdf(pdf_path)

# Combine recognized text
recognized_text = "\n\n".join(page_texts)

# Save OCR text
with open("a01-007u_ocr_trocr.txt", "w", encoding="utf-8") as f:
    f.write(recognized_text)
print("\n✅ OCR text saved to a01-007u_ocr_trocr.txt")

# Load ground truth
ground_truth = read_docx_text(gt_path)
print("\n✅ Ground truth loaded from", gt_path)

# Evaluate
evaluate_text(recognized_text, ground_truth)

# Save final report
with open("a01-007u_ocr_comparison_report_trocr.txt", "w", encoding="utf-8") as f:
    f.write("OCR ACCURACY REPORT (TrOCR - Handwritten)\n")
    f.write("=" * 60 + "\n")
    f.write(recognized_text + "\n\n")
    f.write("=" * 60 + "\nGround Truth:\n")
    f.write(ground_truth)
print("✅ Full report saved to a01-007u_ocr_comparison_report_trocr.txt")
