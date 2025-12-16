import os
import requests
import base64
import io
import time
from PIL import Image
from docx import Document
import re
from difflib import SequenceMatcher
import json
from dotenv import load_dotenv
import math
import numpy as np
from collections import Counter
import nltk
from nltk.util import ngrams

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

load_dotenv()

class QwenVisionAnalyzer:
    def __init__(self, api_key):
        self.api_key = "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"
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "qwen/qwen-2-vl-72b-instruct"  # Best free vision model
        
    def pdf_to_images(self, pdf_path, dpi=200):
        """Convert PDF to images using PyMuPDF"""
        print(f"📄 Converting PDF to images...")
        try:
            import fitz
            
            pdf_document = fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                mat = fitz.Matrix(dpi/72, dpi/72)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("ppm")
                img = Image.open(io.BytesIO(img_data))
                
                jpeg_buffer = io.BytesIO()
                img.save(jpeg_buffer, format='JPEG', quality=95)
                final_image = Image.open(jpeg_buffer)
                images.append(final_image)
                print(f"✅ Page {page_num + 1} converted")
            
            pdf_document.close()
            print(f"🎉 Converted {len(images)} pages total")
            return images
            
        except Exception as e:
            print(f"❌ PDF conversion failed: {e}")
            return []
    
    def image_to_base64(self, image):
        """Convert image to base64 for API"""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode()
    
    def extract_text_from_image(self, image, max_retries=3):
        """Extract text using Qwen Vision model via OpenRouter"""
        for attempt in range(max_retries):
            try:
                img_base64 = self.image_to_base64(image)
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://github.com",
                    "X-Title": "Qwen Handwriting Analyzer"
                }
                
                prompt = """Extract ALL handwritten text exactly as written. 
                Preserve: line breaks, spelling errors, punctuation, capitalization.
                Return ONLY the extracted text, no commentary."""
                
                payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url", 
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 4000
                }
                
                print(f"🔄 Calling Qwen Vision via OpenRouter (attempt {attempt+1})...")
                response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    text = result['choices'][0]['message']['content'].strip()
                    print(f"✅ Qwen extracted {len(text)} characters")
                    return text
                else:
                    print(f"❌ OpenRouter error {response.status_code}: {response.text}")
                    if response.status_code == 429:
                        wait_time = 30 * (attempt + 1)
                        print(f"⏳ Rate limit hit. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        break
                        
            except Exception as e:
                print(f"❌ Attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(30 * (attempt + 1))
                    
        return ""

    def levenshtein_distance(self, s1, s2):
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def calculate_cer(self, extracted_text, ground_truth_text):
        """Calculate Character Error Rate (CER) using Levenshtein distance"""
        extracted_clean = self.preprocess_text(extracted_text)
        ground_truth_clean = self.preprocess_text(ground_truth_text)
        
        distance = self.levenshtein_distance(extracted_clean, ground_truth_clean)
        
        if len(ground_truth_clean) == 0:
            return 100.0
        
        cer = (distance / len(ground_truth_clean)) * 100
        return cer
    
    def calculate_wer(self, extracted_text, ground_truth_text):
        """Calculate Word Error Rate (WER)"""
        extracted_clean = self.preprocess_text(extracted_text)
        ground_truth_clean = self.preprocess_text(ground_truth_text)
        
        extracted_words = extracted_clean.split()
        ground_truth_words = ground_truth_clean.split()
        
        distance = self.levenshtein_distance(extracted_words, ground_truth_words)
        
        if len(ground_truth_words) == 0:
            return 100.0
        
        wer = (distance / len(ground_truth_words)) * 100
        return wer
    
    def calculate_bleu(self, extracted_text, ground_truth_text, n=4):
        """Calculate BLEU score with brevity penalty"""
        extracted_clean = self.preprocess_text(extracted_text)
        ground_truth_clean = self.preprocess_text(ground_truth_text)
        
        candidate_tokens = extracted_clean.split()
        reference_tokens = ground_truth_clean.split()
        
        if len(candidate_tokens) == 0 or len(reference_tokens) == 0:
            return 0.0
        
        c = len(candidate_tokens)
        r = len(reference_tokens)
        
        if c > r:
            bp = 1.0
        else:
            bp = math.exp(1 - r / c) if c > 0 else 0.0
        
        precisions = []
        for i in range(1, n + 1):
            candidate_ngrams = list(ngrams(candidate_tokens, i))
            reference_ngrams = list(ngrams(reference_tokens, i))
            
            if len(candidate_ngrams) == 0:
                precisions.append(0.0)
                continue
                
            match_count = 0
            candidate_counter = Counter(candidate_ngrams)
            reference_counter = Counter(reference_ngrams)
            
            for ngram in candidate_ngrams:
                if ngram in reference_counter:
                    match_count += min(candidate_counter[ngram], reference_counter[ngram])
                    candidate_counter[ngram] = 0
            
            precision = match_count / len(candidate_ngrams) if len(candidate_ngrams) > 0 else 0.0
            precisions.append(precision)
        
        if all(p > 0 for p in precisions):
            geo_mean = math.exp(sum(math.log(p) for p in precisions) / n)
        else:
            geo_mean = 0.0
        
        bleu = bp * geo_mean * 100
        return bleu
    
    def calculate_rouge_n(self, extracted_text, ground_truth_text, n=1):
        """Calculate ROUGE-N score (recall-oriented)"""
        extracted_clean = self.preprocess_text(extracted_text)
        ground_truth_clean = self.preprocess_text(ground_truth_text)
        
        candidate_tokens = extracted_clean.split()
        reference_tokens = ground_truth_clean.split()
        
        if len(reference_tokens) == 0:
            return 0.0
        
        candidate_ngrams = set(ngrams(candidate_tokens, n))
        reference_ngrams = set(ngrams(reference_tokens, n))
        
        overlapping_ngrams = candidate_ngrams & reference_ngrams
        rouge_n = len(overlapping_ngrams) / len(reference_ngrams) * 100 if len(reference_ngrams) > 0 else 0.0
        
        return rouge_n
    
    def longest_common_subsequence(self, seq1, seq2):
        """Calculate length of longest common subsequence"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    def calculate_rouge_l(self, extracted_text, ground_truth_text):
        """Calculate ROUGE-L score based on longest common subsequence"""
        extracted_clean = self.preprocess_text(extracted_text)
        ground_truth_clean = self.preprocess_text(ground_truth_text)
        
        candidate_tokens = extracted_clean.split()
        reference_tokens = ground_truth_clean.split()
        
        if len(reference_tokens) == 0:
            return {'recall': 0.0, 'precision': 0.0, 'f1': 0.0}
        
        lcs_length = self.longest_common_subsequence(candidate_tokens, reference_tokens)
        
        recall = lcs_length / len(reference_tokens) * 100 if len(reference_tokens) > 0 else 0.0
        precision = lcs_length / len(candidate_tokens) * 100 if len(candidate_tokens) > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            'recall': recall,
            'precision': precision,
            'f1': f1
        }
    
    def calculate_bertscore(self, extracted_text, ground_truth_text):
        """Calculate BERTScore for semantic similarity (simplified implementation)"""
        extracted_clean = self.preprocess_text(extracted_text)
        ground_truth_clean = self.preprocess_text(ground_truth_text)
        
        candidate_words = extracted_clean.split()
        reference_words = ground_truth_clean.split()
        
        if not candidate_words or not reference_words:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        all_words = set(candidate_words + reference_words)
        idf_weights = {word: 1.0 for word in all_words}
        
        common_words = set(candidate_words) & set(reference_words)
        
        precision_sum = sum(idf_weights.get(word, 1.0) for word in common_words)
        recall_sum = precision_sum
        
        precision = precision_sum / sum(idf_weights.get(word, 1.0) for word in candidate_words) * 100
        recall = recall_sum / sum(idf_weights.get(word, 1.0) for word in reference_words) * 100
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from all pages of PDF using Qwen Vision"""
        images = self.pdf_to_images(pdf_path)
        extracted_texts = []
        
        for i, image in enumerate(images):
            print(f"📄 Processing page {i+1}/{len(images)}...")
            text = self.extract_text_from_image(image)
            extracted_texts.append(text)
            print(f"✅ Page {i+1} extracted: {len(text)} characters")
            
        return "\n\n".join(extracted_texts)

    def read_docx(self, docx_path):
        """Extract text from DOCX file"""
        doc = Document(docx_path)
        full_text = []
        
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
            
        return "\n".join(full_text)

    def preprocess_text(self, text):
        """Clean and normalize text for comparison"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace but preserve paragraph structure
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Preserve paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces
        text = text.strip()
        
        return text

    def calculate_accuracy(self, extracted_text, ground_truth_text):
        """Calculate various accuracy metrics"""
        
        # Preprocess texts
        extracted_clean = self.preprocess_text(extracted_text)
        ground_truth_clean = self.preprocess_text(ground_truth_text)
        
        # Character-level accuracy
        char_matcher = SequenceMatcher(None, extracted_clean, ground_truth_clean)
        char_similarity = char_matcher.ratio()
        
        # Word-level accuracy
        extracted_words = extracted_clean.split()
        ground_truth_words = ground_truth_clean.split()
        
        word_matcher = SequenceMatcher(None, extracted_words, ground_truth_words)
        word_similarity = word_matcher.ratio()
        
        # Calculate precision, recall, F1 for words
        common_words = set(extracted_words) & set(ground_truth_words)
        
        if len(extracted_words) > 0:
            precision = len(common_words) / len(extracted_words)
        else:
            precision = 0
            
        if len(ground_truth_words) > 0:
            recall = len(common_words) / len(ground_truth_words)
        else:
            recall = 0
            
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0
        
        return {
            'character_accuracy': char_similarity * 100,
            'word_accuracy': word_similarity * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1_score * 100,
            'extracted_word_count': len(extracted_words),
            'ground_truth_word_count': len(ground_truth_words),
            'common_word_count': len(common_words)
        }

    def calculate_all_metrics(self, extracted_text, ground_truth_text):
        """Calculate comprehensive evaluation metrics"""
        
        print("Calculating CER...")
        cer = self.calculate_cer(extracted_text, ground_truth_text)
        
        print("Calculating WER...")
        wer = self.calculate_wer(extracted_text, ground_truth_text)
        
        print("Calculating BLEU scores...")
        bleu_1 = self.calculate_bleu(extracted_text, ground_truth_text, n=1)
        bleu_2 = self.calculate_bleu(extracted_text, ground_truth_text, n=2)
        bleu_3 = self.calculate_bleu(extracted_text, ground_truth_text, n=3)
        bleu_4 = self.calculate_bleu(extracted_text, ground_truth_text, n=4)
        
        print("Calculating ROUGE scores...")
        rouge_1 = self.calculate_rouge_n(extracted_text, ground_truth_text, n=1)
        rouge_2 = self.calculate_rouge_n(extracted_text, ground_truth_text, n=2)
        rouge_l = self.calculate_rouge_l(extracted_text, ground_truth_text)
        
        print("Calculating BERTScore...")
        bertscore = self.calculate_bertscore(extracted_text, ground_truth_text)
        
        print("Calculating traditional accuracy metrics...")
        accuracy_metrics = self.calculate_accuracy(extracted_text, ground_truth_text)
        
        # Combine all metrics
        comprehensive_metrics = {
            # Error Rates
            'cer': cer,
            'wer': wer,
            
            # BLEU Scores
            'bleu_1': bleu_1,
            'bleu_2': bleu_2,
            'bleu_3': bleu_3,
            'bleu_4': bleu_4,
            
            # ROUGE Scores
            'rouge_1': rouge_1,
            'rouge_2': rouge_2,
            'rouge_l_recall': rouge_l['recall'],
            'rouge_l_precision': rouge_l['precision'],
            'rouge_l_f1': rouge_l['f1'],
            
            # BERTScore
            'bertscore_precision': bertscore['precision'],
            'bertscore_recall': bertscore['recall'],
            'bertscore_f1': bertscore['f1'],
            
            # Traditional metrics
            **accuracy_metrics
        }
        
        return comprehensive_metrics

    def detailed_comparison(self, extracted_text, ground_truth_text):
        """Provide detailed comparison with differences"""
        
        extracted_clean = self.preprocess_text(extracted_text)
        ground_truth_clean = self.preprocess_text(ground_truth_text)
        
        extracted_lines = extracted_clean.split('\n')
        ground_truth_lines = ground_truth_clean.split('\n')
        
        differences = []
        
        # Compare line by line
        for i, (ext_line, gt_line) in enumerate(zip(extracted_lines, ground_truth_lines)):
            if ext_line != gt_line:
                differences.append({
                    'line_number': i + 1,
                    'extracted': ext_line,
                    'ground_truth': gt_line,
                    'similarity': SequenceMatcher(None, ext_line, gt_line).ratio() * 100
                })
        
        return differences

    def generate_comprehensive_report(self, extracted_text, ground_truth_text, metrics, differences):
        """Generate a comprehensive evaluation report"""
        
        report = {
            'summary': {
                'character_accuracy': f"{metrics['character_accuracy']:.2f}%",
                'word_accuracy': f"{metrics['word_accuracy']:.2f}%",
                'cer': f"{metrics['cer']:.2f}%",
                'wer': f"{metrics['wer']:.2f}%",
                'bleu_4': f"{metrics['bleu_4']:.2f}%",
                'rouge_l_f1': f"{metrics['rouge_l_f1']:.2f}%",
                'bertscore_f1': f"{metrics['bertscore_f1']:.2f}%",
                'extracted_words': metrics['extracted_word_count'],
                'ground_truth_words': metrics['ground_truth_word_count']
            },
            'detailed_metrics': {
                'error_rates': {
                    'cer': metrics['cer'],
                    'wer': metrics['wer']
                },
                'bleu_scores': {
                    'bleu_1': metrics['bleu_1'],
                    'bleu_2': metrics['bleu_2'],
                    'bleu_3': metrics['bleu_3'],
                    'bleu_4': metrics['bleu_4']
                },
                'rouge_scores': {
                    'rouge_1': metrics['rouge_1'],
                    'rouge_2': metrics['rouge_2'],
                    'rouge_l_recall': metrics['rouge_l_recall'],
                    'rouge_l_precision': metrics['rouge_l_precision'],
                    'rouge_l_f1': metrics['rouge_l_f1']
                },
                'bertscore': {
                    'precision': metrics['bertscore_precision'],
                    'recall': metrics['bertscore_recall'],
                    'f1': metrics['bertscore_f1']
                },
                'traditional_metrics': {
                    'character_accuracy': metrics['character_accuracy'],
                    'word_accuracy': metrics['word_accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score']
                }
            },
            'text_statistics': {
                'extracted_character_count': len(extracted_text),
                'ground_truth_character_count': len(ground_truth_text),
                'extracted_word_count': metrics['extracted_word_count'],
                'ground_truth_word_count': metrics['ground_truth_word_count'],
                'common_word_count': metrics['common_word_count']
            },
            'sample_comparison': {
                'first_100_chars_extracted': extracted_text[:100] + "..." if len(extracted_text) > 100 else extracted_text,
                'first_100_chars_ground_truth': ground_truth_text[:100] + "..." if len(ground_truth_text) > 100 else ground_truth_text
            },
            'major_differences': differences[:10]  # Show first 10 differences
        }
        
        return report


def test_qwen_vision(analyzer):
    """Test Qwen Vision model with a sample image"""
    print("\n🧪 Testing Qwen Vision Model...")
    
    # Create a test image with actual text
    img = Image.new('RGB', (400, 200), color='white')
    from PIL import ImageDraw, ImageFont
    
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((50, 50), "Test Handwriting", fill='black', font=font)
    draw.text((50, 80), "Hello World 123", fill='black', font=font)
    
    result = analyzer.extract_text_from_image(img)
    
    if result and len(result) > 0:
        print(f"🎉 Qwen Vision SUCCESS! Extracted: {result}")
        return True
    else:
        print("❌ Qwen Vision failed to extract text")
        return False


def test_file_processing(analyzer):
    """Test if we can process actual files"""
    print("🧪 Testing file processing...")
    
    # Update these paths to your actual files
    test_pdf_path = "/content/drive/MyDrive/FinalYP_Development/inputs/23138.pdf"  # Your handwritten PDF
    test_docx_path = "/content/drive/MyDrive/FinalYP_Development/inputs/23138.docx"  # Your ground truth DOCX
    
    # Check if files exist
    pdf_exists = os.path.exists(test_pdf_path)
    docx_exists = os.path.exists(test_docx_path)
    
    print(f"📄 PDF file '{test_pdf_path}': {'✅ EXISTS' if pdf_exists else '❌ NOT FOUND'}")
    print(f"📝 DOCX file '{test_docx_path}': {'✅ EXISTS' if docx_exists else '❌ NOT FOUND'}")
    
    if pdf_exists and docx_exists:
        # Test DOCX reading
        ground_truth = analyzer.read_docx(test_docx_path)
        print(f"✅ Ground truth loaded: {len(ground_truth)} characters")
        print(f"📖 Sample: {ground_truth[:100]}...")
        
        # Test PDF conversion (without API calls)
        images = analyzer.pdf_to_images(test_pdf_path)
        print(f"✅ PDF converted to {len(images)} images")
        
        return True, ground_truth, images
    else:
        print("❌ Please check your file paths")
        return False, "", []


def main_analysis():
    """Run complete handwriting analysis"""
    
    # Initialize analyzer
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment variables")
        return
    
    qwen_analyzer = QwenVisionAnalyzer(api_key)
    print("✅ Qwen Vision Analyzer created successfully!")
    
    # Your file paths
    handwritten_pdf_path = "/content/drive/MyDrive/FinalYP_Development/inputs/23138.pdf"
    ground_truth_docx_path = "/content/drive/MyDrive/FinalYP_Development/inputs/23138.docx"
    
    print("🚀 STARTING COMPLETE HANDWRITING ANALYSIS WITH QWEN VISION")
    print("=" * 60)
    
    try:
        # Step 1: Read ground truth
        print("\n📝 STEP 1: Reading ground truth...")
        ground_truth_text = qwen_analyzer.read_docx(ground_truth_docx_path)
        print(f"   ✅ Ground truth: {len(ground_truth_text)} characters")
        print(f"   📖 Sample: {ground_truth_text[:150]}...")
        
        # Step 2: Extract text from handwritten PDF
        print("\n📄 STEP 2: Extracting text from handwritten PDF...")
        extracted_text = qwen_analyzer.extract_text_from_pdf(handwritten_pdf_path)
        print(f"   ✅ Extracted text: {len(extracted_text)} characters")
        print(f"   📖 Sample: {extracted_text[:150]}...")
        
        # Save extracted text to .txt file
        print("\n💾 Saving extracted text to file...")
        text_output_file = "/content/drive/MyDrive/FinalYP_Development/outputs/Qwen/qwen_extracted_text_23138.txt"
        with open(text_output_file, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        print(f"✅ Full extracted text saved to: {text_output_file}")
        
        # Step 3: Calculate comprehensive metrics
        print("\n📊 STEP 3: Calculating comprehensive evaluation metrics...")
        comprehensive_metrics = qwen_analyzer.calculate_all_metrics(extracted_text, ground_truth_text)
        
        # Step 4: Detailed comparison
        print("\n🔍 STEP 4: Performing detailed comparison...")
        differences = qwen_analyzer.detailed_comparison(extracted_text, ground_truth_text)
        
        # Step 5: Generate comprehensive report
        print("\n📈 STEP 5: Generating final report...")
        report = qwen_analyzer.generate_comprehensive_report(extracted_text, ground_truth_text, comprehensive_metrics, differences)
        
        # Display Results
        print("\n" + "=" * 60)
        print("🎯 COMPREHENSIVE EVALUATION RESULTS")
        print("=" * 60)
        
        print(f"\n📊 ACCURACY SUMMARY:")
        print(f"   Character Accuracy: {report['summary']['character_accuracy']}")
        print(f"   Word Accuracy: {report['summary']['word_accuracy']}")
        print(f"   Character Error Rate (CER): {report['summary']['cer']}")
        print(f"   Word Error Rate (WER): {report['summary']['wer']}")
        print(f"   BLEU-4 Score: {report['summary']['bleu_4']}")
        print(f"   ROUGE-L F1: {report['summary']['rouge_l_f1']}")
        print(f"   BERTScore F1: {report['summary']['bertscore_f1']}")
        print(f"   Extracted Words: {report['summary']['extracted_words']}")
        print(f"   Ground Truth Words: {report['summary']['ground_truth_words']}")
        
        print(f"\n📝 TEXT COMPARISON:")
        print(f"   Extracted: {report['sample_comparison']['first_100_chars_extracted']}")
        print(f"   Expected:  {report['sample_comparison']['first_100_chars_ground_truth']}")
        
        if differences:
            print(f"\n❌ MAJOR DIFFERENCES (Top {len(report['major_differences'])}):")
            for diff in report['major_differences']:
                print(f"   Line {diff['line_number']} ({diff['similarity']:.1f}% similar):")
                print(f"      📤 Extracted: {diff['extracted']}")
                print(f"      ✅ Expected:  {diff['ground_truth']}")
                print()
        
        # Save comprehensive results to file
        output_file = "/content/drive/MyDrive/FinalYP_Development/outputs/Qwen/qwen_comprehensive_analysis_23138.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Full results saved to: {output_file}")
        
        # Also save raw metrics for further analysis
        metrics_file = "/content/drive/MyDrive/FinalYP_Development/outputs/Qwen/qwen_raw_metrics_23138.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_metrics, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Raw metrics data saved to: {metrics_file}")
        print("🎉 ANALYSIS COMPLETE!")
        
        return report
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run tests first
    api_key = os.getenv('OPENROUTER_API_KEY')
    if api_key:
        qwen_analyzer = QwenVisionAnalyzer(api_key)
        
        # Test the model
        test_qwen_vision(qwen_analyzer)
        
        # Test file processing
        success, ground_truth, images = test_file_processing(qwen_analyzer)
        
        # Run main analysis if files exist
        if success:
            final_report = main_analysis()
        else:
            print("❌ Cannot run main analysis - required files not found")
    else:
        print("❌ Please set OPENROUTER_API_KEY in your .env file")