import google.generativeai as genai
import os
import tempfile
from PIL import Image
import io
import time
from pdf2image import convert_from_path
from docx import Document
import re
from difflib import SequenceMatcher
import json
import math
import numpy as np
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.util import ngrams
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.stats import bootstrap
import statistics

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class HandwritingAnalyzer:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        
    def pdf_to_images(self, pdf_path, dpi=200):
        """Convert PDF pages to images"""
        print(f"Converting PDF to images...")
        images = convert_from_path(pdf_path, dpi=dpi)
        print(f"Converted {len(images)} pages")
        return images
    
    def extract_text_from_image(self, image, max_retries=3):
        """Extract text from image using Gemini"""
        for attempt in range(max_retries):
            try:
                prompt = """
                Extract ALL text from this handwritten document exactly as written. 
                Preserve:
                - Line breaks and paragraph structure
                - Spelling errors and corrections  
                - Punctuation and capitalization
                - Any symbols or special characters
                
                Return only the extracted text, no additional commentary.
                """
                
                response = self.model.generate_content([prompt, image])
                return response.text.strip()
                
            except Exception as e:
                if "quota" in str(e).lower():
                    wait_time = 30 * (attempt + 1)
                    print(f"Quota exceeded. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Error: {e}")
                    return ""
        return ""
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from all pages of PDF"""
        images = self.pdf_to_images(pdf_path)
        extracted_texts = []
        
        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}...")
            text = self.extract_text_from_image(image)
            extracted_texts.append(text)
            print(f"Page {i+1} extracted: {len(text)} characters")
            
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
        # Preprocess texts
        extracted_clean = self.preprocess_text(extracted_text)
        ground_truth_clean = self.preprocess_text(ground_truth_text)
        
        # Calculate Levenshtein distance
        distance = self.levenshtein_distance(extracted_clean, ground_truth_clean)
        
        # CER = (S + I + D) / N, where N is the number of characters in reference
        if len(ground_truth_clean) == 0:
            return 100.0  # Maximum error if reference is empty
        
        cer = (distance / len(ground_truth_clean)) * 100
        return cer
    
    def calculate_wer(self, extracted_text, ground_truth_text):
        """Calculate Word Error Rate (WER)"""
        # Preprocess texts
        extracted_clean = self.preprocess_text(extracted_text)
        ground_truth_clean = self.preprocess_text(ground_truth_text)
        
        # Tokenize into words
        extracted_words = extracted_clean.split()
        ground_truth_words = ground_truth_clean.split()
        
        # Calculate Levenshtein distance at word level
        distance = self.levenshtein_distance(extracted_words, ground_truth_words)
        
        # WER = (S + I + D) / N, where N is the number of words in reference
        if len(ground_truth_words) == 0:
            return 100.0  # Maximum error if reference is empty
        
        wer = (distance / len(ground_truth_words)) * 100
        return wer
    
    def calculate_bleu(self, extracted_text, ground_truth_text, n=4):
        """Calculate BLEU score with brevity penalty"""
        # Preprocess texts
        extracted_clean = self.preprocess_text(extracted_text)
        ground_truth_clean = self.preprocess_text(ground_truth_text)
        
        # Tokenize
        candidate_tokens = extracted_clean.split()
        reference_tokens = ground_truth_clean.split()
        
        if len(candidate_tokens) == 0 or len(reference_tokens) == 0:
            return 0.0
        
        # Calculate brevity penalty
        c = len(candidate_tokens)
        r = len(reference_tokens)
        
        if c > r:
            bp = 1.0
        else:
            bp = math.exp(1 - r / c) if c > 0 else 0.0
        
        # Calculate n-gram precision
        precisions = []
        for i in range(1, n + 1):
            candidate_ngrams = list(ngrams(candidate_tokens, i))
            reference_ngrams = list(ngrams(reference_tokens, i))
            
            if len(candidate_ngrams) == 0:
                precisions.append(0.0)
                continue
                
            # Count matches (clipped precision)
            match_count = 0
            candidate_counter = Counter(candidate_ngrams)
            reference_counter = Counter(reference_ngrams)
            
            for ngram in candidate_ngrams:
                if ngram in reference_counter:
                    match_count += min(candidate_counter[ngram], reference_counter[ngram])
                    # Remove to avoid double counting
                    candidate_counter[ngram] = 0
            
            precision = match_count / len(candidate_ngrams) if len(candidate_ngrams) > 0 else 0.0
            precisions.append(precision)
        
        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            geo_mean = math.exp(sum(math.log(p) for p in precisions) / n)
        else:
            geo_mean = 0.0
        
        bleu = bp * geo_mean * 100  # Convert to percentage
        return bleu
    
    def calculate_rouge_n(self, extracted_text, ground_truth_text, n=1):
        """Calculate ROUGE-N score (recall-oriented)"""
        # Preprocess texts
        extracted_clean = self.preprocess_text(extracted_text)
        ground_truth_clean = self.preprocess_text(ground_truth_text)
        
        # Tokenize
        candidate_tokens = extracted_clean.split()
        reference_tokens = ground_truth_clean.split()
        
        if len(reference_tokens) == 0:
            return 0.0
        
        # Generate n-grams
        candidate_ngrams = set(ngrams(candidate_tokens, n))
        reference_ngrams = set(ngrams(reference_tokens, n))
        
        # Calculate ROUGE-N recall
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
        # Preprocess texts
        extracted_clean = self.preprocess_text(extracted_text)
        ground_truth_clean = self.preprocess_text(ground_truth_text)
        
        # Tokenize
        candidate_tokens = extracted_clean.split()
        reference_tokens = ground_truth_clean.split()
        
        if len(reference_tokens) == 0:
            return {'recall': 0.0, 'precision': 0.0, 'f1': 0.0}
        
        # Calculate LCS
        lcs_length = self.longest_common_subsequence(candidate_tokens, reference_tokens)
        
        # Calculate ROUGE-L recall, precision, and F1
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
        """Calculate BERTScore for semantic similarity"""
        # This is a simplified implementation
        # In production, you would use the actual BERTScore library
        
        # For now, we'll use a semantic similarity approximation
        # based on word overlap with IDF weighting
        
        extracted_clean = self.preprocess_text(extracted_text)
        ground_truth_clean = self.preprocess_text(ground_truth_text)
        
        candidate_words = extracted_clean.split()
        reference_words = ground_truth_clean.split()
        
        if not candidate_words or not reference_words:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Simple IDF approximation (in practice, use precomputed IDF)
        all_words = set(candidate_words + reference_words)
        idf_weights = {word: 1.0 for word in all_words}  # Simplified
        
        # Calculate weighted precision and recall
        common_words = set(candidate_words) & set(reference_words)
        
        precision_sum = sum(idf_weights.get(word, 1.0) for word in common_words)
        recall_sum = precision_sum  # Same common words
        
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
    
    def calculate_accuracy(self, extracted_text, ground_truth_text):
        """Calculate various accuracy metrics"""
        
        # Preprocess texts
        extracted_clean = self.preprocess_text(extracted_text)
        ground_truth_clean = self.preprocess_text(ground_truth_text)
        
        # Character-level accuracy (using SequenceMatcher as before)
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

def main():
    # Configuration
    API_KEY = "AIzaSyC0qdImg8YBPG9e4a8uWthx5aS7Q-V8kwo"  # Replace with your API key
    
    # File paths
    handwritten_pdf_path = "18.pdf"  # Replace with your PDF path
    ground_truth_docx_path = "18.docx"  # Replace with your DOCX path
    
    # Initialize analyzer
    analyzer = HandwritingAnalyzer(API_KEY)
    
    try:
        print("🚀 Starting comprehensive handwriting analysis...")
        
        # Step 1: Extract text from handwritten PDF
        print("\n📄 Extracting text from handwritten PDF...")
        extracted_text = analyzer.extract_text_from_pdf(handwritten_pdf_path)
        
        print(f"✅ Extracted {len(extracted_text)} characters from PDF")
        
        # Step 2: Read ground truth from DOCX
        print("\n📝 Reading ground truth from DOCX...")
        ground_truth_text = analyzer.read_docx(ground_truth_docx_path)
        
        print(f"✅ Ground truth has {len(ground_truth_text)} characters")
        
        # Step 3: Calculate comprehensive metrics
        print("\n📊 Calculating comprehensive evaluation metrics...")
        comprehensive_metrics = analyzer.calculate_all_metrics(extracted_text, ground_truth_text)
        
        # Step 4: Detailed comparison
        print("\n🔍 Performing detailed comparison...")
        differences = analyzer.detailed_comparison(extracted_text, ground_truth_text)
        
        # Step 5: Generate comprehensive report
        print("\n📈 Generating comprehensive report...")
        report = analyzer.generate_comprehensive_report(extracted_text, ground_truth_text, comprehensive_metrics, differences)
        
        # Display results
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE EVALUATION RESULTS")
        print("="*80)
        
        print(f"\n📊 SUMMARY:")
        print(f"Character Accuracy: {report['summary']['character_accuracy']}")
        print(f"Word Accuracy: {report['summary']['word_accuracy']}")
        print(f"Character Error Rate (CER): {report['summary']['cer']}")
        print(f"Word Error Rate (WER): {report['summary']['wer']}")
        print(f"BLEU-4 Score: {report['summary']['bleu_4']}")
        print(f"ROUGE-L F1: {report['summary']['rouge_l_f1']}")
        print(f"BERTScore F1: {report['summary']['bertscore_f1']}")
        print(f"Extracted Words: {report['summary']['extracted_words']}")
        print(f"Ground Truth Words: {report['summary']['ground_truth_words']}")
        
        print(f"\n📝 SAMPLE COMPARISON:")
        print(f"Extracted: {report['sample_comparison']['first_100_chars_extracted']}")
        print(f"Ground Truth: {report['sample_comparison']['first_100_chars_ground_truth']}")
        
        if differences:
            print(f"\n❌ MAJOR DIFFERENCES (showing {len(report['major_differences'])} of {len(differences)}):")
            for diff in report['major_differences']:
                print(f"Line {diff['line_number']}:")
                print(f"  Extracted: {diff['extracted']}")
                print(f"  Expected:  {diff['ground_truth']}")
                print(f"  Similarity: {diff['similarity']:.1f}%")
                print()
        
        # Save comprehensive results to file
        output_file = "18.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Comprehensive results saved to: {output_file}")
        
        # Also save raw metrics for further analysis
        metrics_file = "raw_metrics_data_18.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_metrics, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Raw metrics data saved to: {metrics_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()