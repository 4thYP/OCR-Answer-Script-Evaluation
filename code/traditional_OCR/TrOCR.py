import re
import torch
from PIL import Image
import fitz  # PyMuPDF
import io
from difflib import SequenceMatcher
import docx
import os
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import bert_score
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import warnings
warnings.filterwarnings('ignore')

# Check if CUDA is available and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load TrOCR model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(device)

def ocr_pdf_trocr(pdf_path, first_page=None, last_page=None, dpi=300):
    """
    Convert PDF to images using PyMuPDF and perform OCR with TrOCR
    """
    # Open the PDF
    doc = fitz.open(pdf_path)

    # Set page range
    start_page = 0 if first_page is None else first_page - 1
    end_page = len(doc) if last_page is None else last_page

    page_texts = []

    for page_num in range(start_page, end_page):
        page = doc.load_page(page_num)

        # Render page as high-quality image
        mat = fitz.Matrix(dpi/72, dpi/72)  # High resolution matrix
        pix = page.get_pixmap(matrix=mat)

        # Convert to PIL Image
        img_data = pix.tobytes("ppm")
        image = Image.open(io.BytesIO(img_data)).convert('RGB')

        # Perform OCR with TrOCR
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        txt = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        page_texts.append(txt)

        print(f"[debug] TrOCR'd Page {page_num + 1}: {len(txt)} chars")

    doc.close()
    return page_texts

def read_docx(file_path):
    """Read text from a Word document"""
    doc = docx.Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)

def preprocess_text(text):
    """
    Preprocess text by removing extra whitespace, normalizing case,
    and removing special characters for better comparison
    """
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,;:!?()\-]', '', text)
    return text.strip()

def calculate_character_error_rate(reference, hypothesis):
    """
    Calculate Character Error Rate (CER) between reference and hypothesis texts
    CER = (S + D + I) / N
    where:
        S = number of character substitutions
        D = number of character deletions
        I = number of character insertions
        N = number of characters in reference
    """
    # Convert to character lists
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)

    # Create a matrix for dynamic programming (Levenshtein distance at character level)
    d = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1))

    # Initialize matrix
    for i in range(len(ref_chars) + 1):
        d[i, 0] = i
    for j in range(len(hyp_chars) + 1):
        d[0, j] = j

    # Fill the matrix
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                d[i, j] = d[i-1, j-1]  # No cost for matching characters
            else:
                substitution = d[i-1, j-1] + 1
                insertion = d[i, j-1] + 1
                deletion = d[i-1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)

    # Backtrack to find the operations
    i, j = len(ref_chars), len(hyp_chars)
    substitutions = 0
    deletions = 0
    insertions = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_chars[i-1] == hyp_chars[j-1]:
            i -= 1
            j -= 1
        else:
            if i > 0 and j > 0 and d[i, j] == d[i-1, j-1] + 1:
                substitutions += 1
                i -= 1
                j -= 1
            elif j > 0 and d[i, j] == d[i, j-1] + 1:
                insertions += 1
                j -= 1
            elif i > 0 and d[i, j] == d[i-1, j] + 1:
                deletions += 1
                i -= 1

    # Calculate CER
    total_chars = len(ref_chars)
    if total_chars == 0:
        return {
            'cer': 0,
            'substitutions': 0,
            'deletions': 0,
            'insertions': 0,
            'total_chars': 0,
            'error_count': 0
        }

    cer = (substitutions + deletions + insertions) / total_chars * 100

    return {
        'cer': cer,
        'substitutions': substitutions,
        'deletions': deletions,
        'insertions': insertions,
        'total_chars': total_chars,
        'error_count': substitutions + deletions + insertions
    }

def calculate_word_error_rate(reference, hypothesis):
    """
    Calculate Word Error Rate (WER) between reference and hypothesis texts
    WER = (S + D + I) / N
    where:
        S = number of word substitutions
        D = number of word deletions
        I = number of word insertions
        N = number of words in reference
    """
    # Tokenize into words
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Create a matrix for dynamic programming (Levenshtein distance at word level)
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))

    # Initialize matrix
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j

    # Fill the matrix
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i, j] = d[i-1, j-1]  # No cost for matching words
            else:
                substitution = d[i-1, j-1] + 1
                insertion = d[i, j-1] + 1
                deletion = d[i-1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)

    # Backtrack to find the operations
    i, j = len(ref_words), len(hyp_words)
    substitutions = 0
    deletions = 0
    insertions = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i-1] == hyp_words[j-1]:
            i -= 1
            j -= 1
        else:
            if i > 0 and j > 0 and d[i, j] == d[i-1, j-1] + 1:
                substitutions += 1
                i -= 1
                j -= 1
            elif j > 0 and d[i, j] == d[i, j-1] + 1:
                insertions += 1
                j -= 1
            elif i > 0 and d[i, j] == d[i-1, j] + 1:
                deletions += 1
                i -= 1

    # Calculate WER
    total_words = len(ref_words)
    if total_words == 0:
        return {
            'wer': 0,
            'substitutions': 0,
            'deletions': 0,
            'insertions': 0,
            'total_words': 0,
            'error_count': 0
        }

    wer = (substitutions + deletions + insertions) / total_words * 100

    return {
        'wer': wer,
        'substitutions': substitutions,
        'deletions': deletions,
        'insertions': insertions,
        'total_words': total_words,
        'error_count': substitutions + deletions + insertions
    }

def calculate_bleu_score(reference, hypothesis):
    """
    Calculate BLEU score between reference and hypothesis texts
    BLEU is a precision-oriented metric that measures n-gram overlap
    """
    # Tokenize into words
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Handle empty edge cases
    if not ref_words or not hyp_words:
        return 0.0

    # Use smoothing function to handle short sentences
    smoothing = SmoothingFunction().method1

    # Calculate BLEU score (using 4-gram with smoothing)
    try:
        bleu_score = sentence_bleu([ref_words], hyp_words,
                                  weights=(0.25, 0.25, 0.25, 0.25),  # 4-gram weights
                                  smoothing_function=smoothing)
    except:
        bleu_score = 0.0

    return bleu_score * 100  # Convert to percentage

def calculate_rouge_scores(reference, hypothesis):
    """
    Calculate ROUGE scores between reference and hypothesis texts
    ROUGE is a recall-oriented metric that measures n-gram overlap
    """
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Calculate scores
    scores = scorer.score(reference, hypothesis)

    # Extract F1 scores (harmonic mean of precision and recall)
    rouge1 = scores['rouge1'].fmeasure * 100
    rouge2 = scores['rouge2'].fmeasure * 100
    rougeL = scores['rougeL'].fmeasure * 100

    return {
        'rouge1': rouge1,
        'rouge2': rouge2,
        'rougeL': rougeL
    }

def calculate_bertscore(reference, hypothesis):
    """
    Calculate BERTScore between reference and hypothesis texts
    BERTScore measures semantic similarity using contextual embeddings
    """
    try:
        # Calculate BERTScore
        P, R, F1 = bert_score.score([hypothesis], [reference], lang="en", verbose=False)

        # Return F1 score (harmonic mean of precision and recall)
        return F1.item() * 100
    except:
        return 0.0

def highlight_word_differences(reference, hypothesis):
    """
    Generate a text with highlights showing word-level differences
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Use SequenceMatcher for word-level comparison
    matcher = SequenceMatcher(None, ref_words, hyp_words)

    result = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            result.extend(ref_words[i1:i2])
        elif tag == 'replace':
            # Substitution
            result.append(f"[SUB: {' '.join(ref_words[i1:i2])} -> {' '.join(hyp_words[j1:j2])}]")
        elif tag == 'delete':
            # Deletion
            result.append(f"[DEL: {' '.join(ref_words[i1:i2])}]")
        elif tag == 'insert':
            # Insertion
            result.append(f"[INS: {' '.join(hyp_words[j1:j2])}]")

    return ' '.join(result)

def main():
    # Perform OCR on the PDF using TrOCR
    pdf_file = "a01-007u.pdf"
    print(f"Performing OCR on {pdf_file} using TrOCR...")
    ocr_pages = ocr_pdf_trocr(pdf_file)

    # Read ground truth from Word document
    gt_file = "a01-007u.docx"
    print(f"Reading ground truth from {gt_file}...")
    ground_truth = read_docx(gt_file)

    # Save OCR results
    full_ocr_text = "".join(f"\n\n--- Page {i} ---\n{t}" for i, t in enumerate(ocr_pages, 1))
    with open("a01-007u_trocr.txt", "w", encoding="utf-8") as f:
        f.write(full_ocr_text)
    print("\nSaved full TrOCR results to a01-007u_trocr.txt")

    print("\n" + "="*80)
    print("TrOCR ACCURACY ANALYSIS REPORT")
    print("="*80)

    # Preprocess texts for metric calculations
    gt_clean = preprocess_text(ground_truth)

    # Calculate metrics for each page
    page_cer_metrics = []
    page_wer_metrics = []
    page_bleu_scores = []
    page_rouge_scores = []
    page_bertscores = []

    for i, page_text in enumerate(ocr_pages, 1):
        # Preprocess OCR text
        ocr_clean = preprocess_text(page_text)

        # Character Error Rate (CER)
        cer_metrics = calculate_character_error_rate(gt_clean, ocr_clean)
        page_cer_metrics.append(cer_metrics)

        # Word Error Rate (WER)
        wer_metrics = calculate_word_error_rate(gt_clean, ocr_clean)
        page_wer_metrics.append(wer_metrics)

        # BLEU score
        bleu_score = calculate_bleu_score(gt_clean, ocr_clean)
        page_bleu_scores.append(bleu_score)

        # ROUGE scores
        rouge_scores = calculate_rouge_scores(gt_clean, ocr_clean)
        page_rouge_scores.append(rouge_scores)

        # BERTScore
        bertscore = calculate_bertscore(gt_clean, ocr_clean)
        page_bertscores.append(bertscore)

        print(f"Page {i}:")
        print(f"  Character Error Rate (CER) = {cer_metrics['cer']:.2f}%")
        print(f"  Word Error Rate (WER) = {wer_metrics['wer']:.2f}%")
        print(f"  BLEU Score = {bleu_score:.2f}%")
        print(f"  ROUGE-1 = {rouge_scores['rouge1']:.2f}%")
        print(f"  ROUGE-2 = {rouge_scores['rouge2']:.2f}%")
        print(f"  ROUGE-L = {rouge_scores['rougeL']:.2f}%")
        print(f"  BERTScore = {bertscore:.2f}%")
        print(f"  Character Errors: {cer_metrics['error_count']} chars (S:{cer_metrics['substitutions']}, D:{cer_metrics['deletions']}, I:{cer_metrics['insertions']})")
        print(f"  Word Errors: {wer_metrics['error_count']} words (S:{wer_metrics['substitutions']}, D:{wer_metrics['deletions']}, I:{wer_metrics['insertions']})")
        print()

    # Calculate overall metrics
    combined_ocr = " ".join(ocr_pages)
    combined_ocr_clean = preprocess_text(combined_ocr)

    overall_cer_metrics = calculate_character_error_rate(gt_clean, combined_ocr_clean)
    overall_wer_metrics = calculate_word_error_rate(gt_clean, combined_ocr_clean)
    overall_bleu = calculate_bleu_score(gt_clean, combined_ocr_clean)
    overall_rouge = calculate_rouge_scores(gt_clean, combined_ocr_clean)
    overall_bertscore = calculate_bertscore(gt_clean, combined_ocr_clean)

    print(f"\nOverall Results:")
    print(f"Character Error Rate (CER) = {overall_cer_metrics['cer']:.2f}%")
    print(f"Word Error Rate (WER) = {overall_wer_metrics['wer']:.2f}%")
    print(f"BLEU Score = {overall_bleu:.2f}%")
    print(f"ROUGE-1 = {overall_rouge['rouge1']:.2f}%")
    print(f"ROUGE-2 = {overall_rouge['rouge2']:.2f}%")
    print(f"ROUGE-L = {overall_rouge['rougeL']:.2f}%")
    print(f"BERTScore = {overall_bertscore:.2f}%")
    print(f"Total Character Errors: {overall_cer_metrics['error_count']} chars (S:{overall_cer_metrics['substitutions']}, D:{overall_cer_metrics['deletions']}, I:{overall_cer_metrics['insertions']})")
    print(f"Total Word Errors: {overall_wer_metrics['error_count']} words (S:{overall_wer_metrics['substitutions']}, D:{overall_wer_metrics['deletions']}, I:{overall_wer_metrics['insertions']})")
    print(f"Total characters in reference: {overall_cer_metrics['total_chars']}")
    print(f"Total words in reference: {overall_wer_metrics['total_words']}")

    # Show detailed differences for the first page
    print("\n" + "="*80)
    print("WORD-LEVEL DIFFERENCES (Page 1)")
    print("="*80)
    if page_cer_metrics:
        diff_text = highlight_word_differences(gt_clean, preprocess_text(ocr_pages[0]))
        print(diff_text[:1000] + "..." if len(diff_text) > 1000 else diff_text)

    # Save detailed comparison to file
    with open("a01-007u_trocr_comprehensive_report.txt", "w", encoding="utf-8") as f:
        f.write("TrOCR COMPREHENSIVE ACCURACY ANALYSIS REPORT\n")
        f.write("="*80 + "\n")

        f.write("\nCHARACTER ERROR RATE (CER):\n")
        f.write("-" * 50 + "\n")
        for i, metrics in enumerate(page_cer_metrics, 1):
            f.write(f"Page {i}: CER = {metrics['cer']:.2f}%, Errors: {metrics['error_count']} (S:{metrics['substitutions']}, D:{metrics['deletions']}, I:{metrics['insertions']})\n")

        f.write("\nWORD ERROR RATE (WER):\n")
        f.write("-" * 50 + "\n")
        for i, metrics in enumerate(page_wer_metrics, 1):
            f.write(f"Page {i}: WER = {metrics['wer']:.2f}%, Errors: {metrics['error_count']} (S:{metrics['substitutions']}, D:{metrics['deletions']}, I:{metrics['insertions']})\n")

        f.write("\nBLEU SCORES (n-gram precision):\n")
        f.write("-" * 50 + "\n")
        for i, score in enumerate(page_bleu_scores, 1):
            f.write(f"Page {i}: BLEU = {score:.2f}%\n")

        f.write("\nROUGE SCORES (n-gram recall):\n")
        f.write("-" * 50 + "\n")
        for i, scores in enumerate(page_rouge_scores, 1):
            f.write(f"Page {i}: ROUGE-1 = {scores['rouge1']:.2f}%, ROUGE-2 = {scores['rouge2']:.2f}%, ROUGE-L = {scores['rougeL']:.2f}%\n")

        f.write("\nBERTSCORES (semantic similarity):\n")
        f.write("-" * 50 + "\n")
        for i, score in enumerate(page_bertscores, 1):
            f.write(f"Page {i}: BERTScore = {score:.2f}%\n")

        f.write(f"\nOVERALL RESULTS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Character Error Rate (CER) = {overall_cer_metrics['cer']:.2f}%\n")
        f.write(f"Word Error Rate (WER) = {overall_wer_metrics['wer']:.2f}%\n")
        f.write(f"BLEU Score = {overall_bleu:.2f}%\n")
        f.write(f"ROUGE-1 = {overall_rouge['rouge1']:.2f}%\n")
        f.write(f"ROUGE-2 = {overall_rouge['rouge2']:.2f}%\n")
        f.write(f"ROUGE-L = {overall_rouge['rougeL']:.2f}%\n")
        f.write(f"BERTScore = {overall_bertscore:.2f}%\n")
        f.write(f"Total Character Errors: {overall_cer_metrics['error_count']} chars (S:{overall_cer_metrics['substitutions']}, D:{overall_cer_metrics['deletions']}, I:{overall_cer_metrics['insertions']})")
        f.write(f"Total Word Errors: {overall_wer_metrics['error_count']} words (S:{overall_wer_metrics['substitutions']}, D:{overall_wer_metrics['deletions']}, I:{overall_wer_metrics['insertions']})")
        f.write(f"Total characters in reference: {overall_cer_metrics['total_chars']}\n")
        f.write(f"Total words in reference: {overall_wer_metrics['total_words']}\n\n")

        f.write("WORD-LEVEL DIFFERENCES\n")
        f.write("="*80 + "\n")
        for i, (cer_metrics, wer_metrics) in enumerate(zip(page_cer_metrics, page_wer_metrics), 1):
            f.write(f"\n--- Page {i} Differences ---\n")
            f.write(f"CER: {cer_metrics['cer']:.2f}%, WER: {wer_metrics['wer']:.2f}%\n")
            diff = highlight_word_differences(gt_clean, preprocess_text(ocr_pages[i-1]))
            f.write(diff + "\n")

    print("\nSaved detailed comparison report to a01-007u_trocr_comprehensive_report.txt")

if __name__ == "__main__":
    main()