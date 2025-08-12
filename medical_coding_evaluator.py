#!/usr/bin/env python3
"""
Medical Coding Evaluation System
===============================

Implements comprehensive evaluation metrics for medical coding systems following medHELM methodology.
Compares agent predictions against ground truth codes with detailed statistical analysis.

Key Features:
- Agent prediction capture and storage
- Multi-level evaluation metrics (exact match, partial match, clinical relevance)
- Statistical analysis with precision, recall, F1 scores
- Code-level performance tracking
- Confidence score analysis
- Export capabilities for research analysis

Usage:
    from medical_coding_evaluator import MedicalCodingEvaluator
    
    evaluator = MedicalCodingEvaluator(dataset_path, rag_system)
    results = evaluator.run_comprehensive_evaluation()
"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import logging
from dataclasses import dataclass
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Structure for agent prediction results."""
    codes: List[str]
    descriptions: List[str]
    confidence_scores: List[float]
    reasoning: str
    timestamp: str
    model_used: str

@dataclass
class EvaluationMetrics:
    """Structure for evaluation metrics."""
    exact_match: float
    partial_match: float
    precision: float
    recall: float
    f1_score: float
    code_level_accuracy: Dict[str, float]
    confidence_correlation: float
    total_predictions: int
    total_ground_truth: int

class MedicalCodingEvaluator:
    """
    Comprehensive medical coding evaluation system.
    
    Evaluates agent performance against ground truth using multiple metrics
    following medHELM paper methodology.
    """
    
    def __init__(self, dataset_path: str, rag_system=None):
        """
        Initialize evaluator with dataset and RAG system.
        
        Args:
            dataset_path: Path to structured medical coding dataset
            rag_system: Initialized RAG system for agent predictions
        """
        self.dataset_path = Path(dataset_path)
        self.rag_system = rag_system
        self.dataset = None
        self.evaluation_results = {}
        
        logger.info(f"Initialized Medical Coding Evaluator with dataset: {dataset_path}")
        
    def load_dataset(self) -> bool:
        """Load the structured medical coding dataset."""
        try:
            if not self.dataset_path.exists():
                logger.error(f"Dataset file not found: {self.dataset_path}")
                return False
                
            with open(self.dataset_path, 'r') as f:
                self.dataset = json.load(f)
                
            logger.info(f"Loaded dataset with {len(self.dataset['records'])} records")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return False
    
    def extract_codes_from_response(self, response: str) -> Tuple[List[str], List[str], List[float]]:
        """
        Extract ICD codes, descriptions, and confidence scores from agent response.
        
        Args:
            response: Raw agent response text
            
        Returns:
            Tuple of (codes, descriptions, confidence_scores)
        """
        codes = []
        descriptions = []
        confidence_scores = []
        
        try:
            # Pattern to match ICD codes (various formats)
            icd_patterns = [
                r'([A-Z]\d{2,3}(?:\.\d{1,2})?)',  # Standard ICD-10 format
                r'ICD-10[:\s]*([A-Z]\d{2,3}(?:\.\d{1,2})?)',  # With ICD-10 prefix
                r'([A-Z]\d{2,3})',  # Without decimal
            ]
            
            # Pattern to match confidence scores
            confidence_pattern = r'confidence[:\s]*(\d+(?:\.\d+)?)'
            
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and headers
                if not line or line.startswith('=') or line.startswith('-'):
                    continue
                
                # Extract codes
                for pattern in icd_patterns:
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    for match in matches:
                        code = match.upper().replace('.', '')  # Normalize format
                        if code not in codes:  # Avoid duplicates
                            codes.append(code)
                            
                            # Extract description (text after the code)
                            desc_match = re.search(f'{re.escape(match)}[:\-\s]*(.+?)(?:confidence|$)', line, re.IGNORECASE)
                            if desc_match:
                                desc = desc_match.group(1).strip()
                                descriptions.append(desc)
                            else:
                                descriptions.append("No description found")
                
                # Extract confidence scores
                conf_matches = re.findall(confidence_pattern, line, re.IGNORECASE)
                for conf in conf_matches:
                    try:
                        score = float(conf)
                        if score > 10:  # Convert percentage to 0-10 scale
                            score = score / 10
                        confidence_scores.append(score)
                    except ValueError:
                        continue
            
            # Ensure equal lengths (pad with defaults if needed)
            max_len = max(len(codes), len(descriptions), len(confidence_scores))
            
            while len(codes) < max_len:
                codes.append("UNKNOWN")
            while len(descriptions) < max_len:
                descriptions.append("No description")
            while len(confidence_scores) < max_len:
                confidence_scores.append(5.0)  # Default medium confidence
            
            # Trim to shortest list to ensure consistency
            min_len = min(len(codes), len(descriptions), len(confidence_scores))
            codes = codes[:min_len]
            descriptions = descriptions[:min_len]
            confidence_scores = confidence_scores[:min_len]
            
            logger.debug(f"Extracted {len(codes)} codes from response")
            
        except Exception as e:
            logger.error(f"Error extracting codes from response: {e}")
            
        return codes, descriptions, confidence_scores
    
    def normalize_code(self, code: str) -> str:
        """Normalize ICD code format for comparison."""
        if not code:
            return ""
        
        # Remove common prefixes and normalize
        code = str(code).upper().strip()
        code = re.sub(r'^ICD-?\d*[:\-\s]*', '', code)  # Remove ICD prefix
        code = re.sub(r'[.\-\s]', '', code)  # Remove separators
        
        return code
    
    def extract_ground_truth_codes(self, record: Dict[str, Any]) -> List[str]:
        """Extract ground truth codes from a record."""
        ground_truth = []
        
        try:
            structured = record.get('structured_findings', {})
            
            # Extract from all finding types
            for finding_type in ['Primary findings', 'Secondary findings', 'Procedures']:
                findings = structured.get(finding_type, [])
                for finding in findings:
                    code = finding.get('code', '')
                    if code:
                        normalized = self.normalize_code(code)
                        if normalized and normalized not in ground_truth:
                            ground_truth.append(normalized)
            
            # Also extract from original icd_codes field as backup
            if 'icd_codes' in record and record['icd_codes']:
                original_codes = str(record['icd_codes']).split(',')
                for code in original_codes:
                    normalized = self.normalize_code(code)
                    if normalized and normalized not in ground_truth:
                        ground_truth.append(normalized)
                        
        except Exception as e:
            logger.error(f"Error extracting ground truth codes: {e}")
            
        return ground_truth
    
    def get_agent_prediction(self, record: Dict[str, Any]) -> PredictionResult:
        """Get agent prediction for a medical record."""
        try:
            # Create medical coding prompt
            note_text = record.get('note_text', '')
            if not note_text:
                logger.warning(f"No note text found for record {record.get('hadm_id')}")
                return PredictionResult([], [], [], "No note text available", 
                                     datetime.now().isoformat(), "unknown")
            
            prompt = f"""
You are a professional medical coder with access to a comprehensive ICD-10 knowledge base.

Analyze this discharge note and provide appropriate ICD-10 codes:

DISCHARGE NOTE:
{note_text}

Please provide:
1. Primary and secondary diagnoses with ICD-10 codes (use exact format: e.g., I120 not I12.0)
2. Procedure codes if applicable
3. Brief explanation for each code selection
4. Confidence level (1-10) for each code

Use your search tools to find accurate codes from the knowledge base.
"""
            
            # Get agent response
            if self.rag_system and hasattr(self.rag_system, 'agent'):
                response = self.rag_system.agent.query(prompt, stream=False)
                
                # Extract structured information from response
                codes, descriptions, confidence_scores = self.extract_codes_from_response(response)
                
                return PredictionResult(
                    codes=codes,
                    descriptions=descriptions,
                    confidence_scores=confidence_scores,
                    reasoning=response,
                    timestamp=datetime.now().isoformat(),
                    model_used="rag_agent"
                )
            else:
                logger.error("RAG system not available for predictions")
                return PredictionResult([], [], [], "RAG system unavailable", 
                                     datetime.now().isoformat(), "none")
                
        except Exception as e:
            logger.error(f"Error getting agent prediction: {e}")
            return PredictionResult([], [], [], f"Error: {str(e)}", 
                                 datetime.now().isoformat(), "error")
    
    def calculate_metrics(self, predicted_codes: List[str], ground_truth_codes: List[str]) -> Dict[str, float]:
        """Calculate evaluation metrics for code predictions."""
        # Normalize codes for comparison
        pred_normalized = [self.normalize_code(code) for code in predicted_codes]
        gt_normalized = [self.normalize_code(code) for code in ground_truth_codes]
        
        # Remove empty codes
        pred_normalized = [code for code in pred_normalized if code]
        gt_normalized = [code for code in gt_normalized if code]
        
        # Convert to sets for comparison
        pred_set = set(pred_normalized)
        gt_set = set(gt_normalized)
        
        # Calculate metrics
        if not gt_set and not pred_set:
            # Both empty - perfect match
            return {
                'exact_match': 1.0,
                'partial_match': 1.0,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0,
                'intersection_size': 0,
                'union_size': 0
            }
        elif not gt_set:
            # No ground truth, but predictions exist
            return {
                'exact_match': 0.0,
                'partial_match': 0.0,
                'precision': 0.0,
                'recall': 0.0 if not pred_set else 1.0,
                'f1_score': 0.0,
                'intersection_size': 0,
                'union_size': len(pred_set)
            }
        elif not pred_set:
            # Ground truth exists, but no predictions
            return {
                'exact_match': 0.0,
                'partial_match': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'intersection_size': 0,
                'union_size': len(gt_set)
            }
        
        # Calculate overlap
        intersection = pred_set.intersection(gt_set)
        union = pred_set.union(gt_set)
        
        # Exact match: perfect overlap
        exact_match = 1.0 if pred_set == gt_set else 0.0
        
        # Partial match: Jaccard similarity
        partial_match = len(intersection) / len(union) if union else 0.0
        
        # Precision: correctly predicted / total predicted
        precision = len(intersection) / len(pred_set) if pred_set else 0.0
        
        # Recall: correctly predicted / total ground truth
        recall = len(intersection) / len(gt_set) if gt_set else 0.0
        
        # F1 score: harmonic mean of precision and recall
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'exact_match': exact_match,
            'partial_match': partial_match,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'intersection_size': len(intersection),
            'union_size': len(union)
        }
    
    def run_evaluation_on_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Run evaluation on a single record."""
        hadm_id = record.get('hadm_id', 'unknown')
        
        try:
            # Get ground truth codes
            ground_truth = self.extract_ground_truth_codes(record)
            
            # Get agent prediction
            prediction = self.get_agent_prediction(record)
            
            # Calculate metrics
            metrics = self.calculate_metrics(prediction.codes, ground_truth)
            
            # Create evaluation result
            result = {
                'hadm_id': hadm_id,
                'ground_truth_codes': ground_truth,
                'predicted_codes': prediction.codes,
                'prediction_descriptions': prediction.descriptions,
                'confidence_scores': prediction.confidence_scores,
                'agent_reasoning': prediction.reasoning,
                'timestamp': prediction.timestamp,
                'model_used': prediction.model_used,
                'metrics': metrics,
                'evaluation_summary': {
                    'total_gt_codes': len(ground_truth),
                    'total_predicted_codes': len(prediction.codes),
                    'correct_predictions': metrics['intersection_size'],
                    'missed_codes': len(ground_truth) - metrics['intersection_size'],
                    'false_positives': len(prediction.codes) - metrics['intersection_size']
                }
            }
            
            logger.info(f"Evaluated record {hadm_id}: F1={metrics['f1_score']:.3f}, "
                       f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating record {hadm_id}: {e}")
            return {
                'hadm_id': hadm_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_comprehensive_evaluation(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on the dataset.
        
        Args:
            sample_size: Number of records to evaluate (None for all)
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Starting comprehensive medical coding evaluation...")
        
        # Load dataset if needed
        if not self.dataset:
            if not self.load_dataset():
                raise ValueError("Failed to load dataset")
        
        records = self.dataset['records']
        
        # Sample if requested
        if sample_size and sample_size < len(records):
            import random
            records = random.sample(records, sample_size)
            logger.info(f"Evaluating sample of {sample_size} records")
        else:
            logger.info(f"Evaluating all {len(records)} records")
        
        # Run evaluation on each record
        evaluation_results = []
        for i, record in enumerate(records):
            logger.info(f"Evaluating record {i+1}/{len(records)}: {record.get('hadm_id')}")
            
            result = self.run_evaluation_on_record(record)
            evaluation_results.append(result)
            
            # Progress update every 10 records
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(records)} records evaluated")
        
        # Calculate aggregate metrics
        aggregate_metrics = self.calculate_aggregate_metrics(evaluation_results)
        
        # Create comprehensive report
        final_results = {
            'evaluation_metadata': {
                'dataset_path': str(self.dataset_path),
                'evaluation_date': datetime.now().isoformat(),
                'total_records_evaluated': len(evaluation_results),
                'sample_size': sample_size,
                'rag_system_info': self.get_rag_system_info()
            },
            'aggregate_metrics': aggregate_metrics,
            'individual_results': evaluation_results,
            'detailed_analysis': self.generate_detailed_analysis(evaluation_results)
        }
        
        # Save results
        self.evaluation_results = final_results
        
        logger.info("Comprehensive evaluation completed!")
        self.print_evaluation_summary(aggregate_metrics)
        
        return final_results
    
    def calculate_aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate metrics across all evaluated records."""
        valid_results = [r for r in results if 'metrics' in r and 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid results to aggregate'}
        
        # Collect individual metrics
        exact_matches = [r['metrics']['exact_match'] for r in valid_results]
        partial_matches = [r['metrics']['partial_match'] for r in valid_results]
        precisions = [r['metrics']['precision'] for r in valid_results]
        recalls = [r['metrics']['recall'] for r in valid_results]
        f1_scores = [r['metrics']['f1_score'] for r in valid_results]
        
        # Calculate totals for micro-averaging
        total_predicted = sum(r['evaluation_summary']['total_predicted_codes'] for r in valid_results)
        total_ground_truth = sum(r['evaluation_summary']['total_gt_codes'] for r in valid_results)
        total_correct = sum(r['evaluation_summary']['correct_predictions'] for r in valid_results)
        
        # Micro-averaged metrics
        micro_precision = total_correct / total_predicted if total_predicted > 0 else 0.0
        micro_recall = total_correct / total_ground_truth if total_ground_truth > 0 else 0.0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        
        # Macro-averaged metrics
        macro_precision = np.mean(precisions) if precisions else 0.0
        macro_recall = np.mean(recalls) if recalls else 0.0
        macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
        
        return {
            'macro_averaged': {
                'exact_match_rate': np.mean(exact_matches),
                'partial_match_rate': np.mean(partial_matches),
                'precision': macro_precision,
                'recall': macro_recall,
                'f1_score': macro_f1
            },
            'micro_averaged': {
                'precision': micro_precision,
                'recall': micro_recall,
                'f1_score': micro_f1
            },
            'totals': {
                'total_records': len(valid_results),
                'total_predicted_codes': total_predicted,
                'total_ground_truth_codes': total_ground_truth,
                'total_correct_predictions': total_correct,
                'total_false_positives': total_predicted - total_correct,
                'total_false_negatives': total_ground_truth - total_correct
            },
            'distributions': {
                'precision_std': np.std(precisions) if precisions else 0.0,
                'recall_std': np.std(recalls) if recalls else 0.0,
                'f1_std': np.std(f1_scores) if f1_scores else 0.0,
                'exact_match_count': sum(exact_matches),
                'partial_match_mean': np.mean(partial_matches) if partial_matches else 0.0
            }
        }
    
    def generate_detailed_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed analysis of evaluation results."""
        valid_results = [r for r in results if 'metrics' in r and 'error' not in r]
        
        # Code frequency analysis
        all_predicted_codes = []
        all_ground_truth_codes = []
        
        for result in valid_results:
            all_predicted_codes.extend(result['predicted_codes'])
            all_ground_truth_codes.extend(result['ground_truth_codes'])
        
        # Most common codes
        from collections import Counter
        predicted_freq = Counter(all_predicted_codes)
        ground_truth_freq = Counter(all_ground_truth_codes)
        
        return {
            'code_analysis': {
                'unique_predicted_codes': len(set(all_predicted_codes)),
                'unique_ground_truth_codes': len(set(all_ground_truth_codes)),
                'most_common_predicted': predicted_freq.most_common(10),
                'most_common_ground_truth': ground_truth_freq.most_common(10)
            },
            'performance_distribution': {
                'high_performance_records': len([r for r in valid_results if r['metrics']['f1_score'] >= 0.8]),
                'medium_performance_records': len([r for r in valid_results if 0.5 <= r['metrics']['f1_score'] < 0.8]),
                'low_performance_records': len([r for r in valid_results if r['metrics']['f1_score'] < 0.5])
            },
            'error_analysis': {
                'records_with_errors': len([r for r in results if 'error' in r]),
                'records_with_no_predictions': len([r for r in valid_results if len(r['predicted_codes']) == 0]),
                'records_with_no_ground_truth': len([r for r in valid_results if len(r['ground_truth_codes']) == 0])
            }
        }
    
    def get_rag_system_info(self) -> Dict[str, Any]:
        """Get information about the RAG system used for evaluation."""
        if not self.rag_system:
            return {'status': 'no_rag_system'}
        
        try:
            info = {
                'system_available': True,
                'agent_available': hasattr(self.rag_system, 'agent') and self.rag_system.agent is not None,
                'knowledge_base_available': hasattr(self.rag_system, 'knowledge_base') and self.rag_system.knowledge_base is not None
            }
            
            # Get additional details if available
            if hasattr(self.rag_system, 'agent') and self.rag_system.agent:
                agent_info = self.rag_system.agent.get_agent_info()
                info['agent_info'] = agent_info
            
            if hasattr(self.rag_system, 'knowledge_base') and self.rag_system.knowledge_base:
                kb_info = self.rag_system.knowledge_base.get_collection_info()
                info['knowledge_base_info'] = kb_info
            
            return info
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def print_evaluation_summary(self, metrics: Dict[str, Any]):
        """Print a formatted summary of evaluation results."""
        print("\n" + "="*60)
        print("🎯 MEDICAL CODING EVALUATION SUMMARY")
        print("="*60)
        
        macro = metrics['macro_averaged']
        micro = metrics['micro_averaged']
        totals = metrics['totals']
        
        print(f"📊 Dataset Overview:")
        print(f"   Total Records Evaluated: {totals['total_records']}")
        print(f"   Total Ground Truth Codes: {totals['total_ground_truth_codes']}")
        print(f"   Total Predicted Codes: {totals['total_predicted_codes']}")
        print(f"   Correct Predictions: {totals['total_correct_predictions']}")
        
        print(f"\n🎯 Macro-Averaged Performance:")
        print(f"   Exact Match Rate: {macro['exact_match_rate']:.3f}")
        print(f"   Precision: {macro['precision']:.3f}")
        print(f"   Recall: {macro['recall']:.3f}")
        print(f"   F1 Score: {macro['f1_score']:.3f}")
        
        print(f"\n📈 Micro-Averaged Performance:")
        print(f"   Precision: {micro['precision']:.3f}")
        print(f"   Recall: {micro['recall']:.3f}")
        print(f"   F1 Score: {micro['f1_score']:.3f}")
        
        print(f"\n🔍 Error Analysis:")
        print(f"   False Positives: {totals['total_false_positives']}")
        print(f"   False Negatives: {totals['total_false_negatives']}")
        
        print("="*60)
    
    def save_results(self, output_path: str):
        """Save evaluation results to file."""
        if not self.evaluation_results:
            logger.error("No evaluation results to save")
            return False
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.evaluation_results, f, indent=2, default=str)
            
            logger.info(f"Evaluation results saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return False
    
    def export_to_csv(self, output_path: str):
        """Export results to CSV for analysis."""
        if not self.evaluation_results or 'individual_results' not in self.evaluation_results:
            logger.error("No evaluation results to export")
            return False
        
        try:
            results = self.evaluation_results['individual_results']
            valid_results = [r for r in results if 'metrics' in r]
            
            # Create DataFrame
            data = []
            for result in valid_results:
                row = {
                    'hadm_id': result['hadm_id'],
                    'ground_truth_count': len(result['ground_truth_codes']),
                    'predicted_count': len(result['predicted_codes']),
                    'correct_predictions': result['metrics']['intersection_size'],
                    'exact_match': result['metrics']['exact_match'],
                    'partial_match': result['metrics']['partial_match'],
                    'precision': result['metrics']['precision'],
                    'recall': result['metrics']['recall'],
                    'f1_score': result['metrics']['f1_score'],
                    'ground_truth_codes': ','.join(result['ground_truth_codes']),
                    'predicted_codes': ','.join(result['predicted_codes'])
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Results exported to CSV: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            return False

def main():
    """Demo function for testing the evaluator."""
    print("🧪 Medical Coding Evaluator - Demo Mode")
    
    # This would normally be called with actual dataset and RAG system
    dataset_path = "/home/justjosh/Turing-Test/mimic-eval/data/enhanced_mimic_iv_structured_coding.json"
    
    if Path(dataset_path).exists():
        evaluator = MedicalCodingEvaluator(dataset_path)
        
        if evaluator.load_dataset():
            print("✅ Dataset loaded successfully")
            print(f"📊 Records available: {len(evaluator.dataset['records'])}")
            
            # Demo: Extract ground truth from first record
            sample_record = evaluator.dataset['records'][0]
            ground_truth = evaluator.extract_ground_truth_codes(sample_record)
            print(f"📋 Sample ground truth codes: {ground_truth}")
        else:
            print("❌ Failed to load dataset")
    else:
        print(f"❌ Dataset not found: {dataset_path}")

if __name__ == "__main__":
    main()
