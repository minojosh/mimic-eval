"""
MIMIC Dataset Optimization Strategies

This script provides several optimization approaches for the MIMIC medical coding dataset
to improve performance while maintaining medical accuracy.
"""

import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

class MIMICDatasetOptimizer:
    """Optimize MIMIC dataset for efficient processing while preserving medical content."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.load_dataset()
        
    def load_dataset(self):
        """Load the MIMIC dataset."""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.records = self.data.get('records', [])
        print(f"Loaded {len(self.records)} records from {self.dataset_path}")
    
    def analyze_note_structure(self) -> Dict:
        """Analyze the structure of discharge summaries to identify optimization opportunities."""
        section_patterns = {
            'header': r'Name:\s*___.*?Sex:\s*[MF]',
            'allergies': r'Allergies:.*?(?=Attending:|Chief Complaint:)',
            'chief_complaint': r'Chief Complaint:.*?(?=Major Surgical|History of Present)',
            'procedures': r'Major Surgical or Invasive Procedure:.*?(?=History of Present|Past Medical)',
            'history_present': r'History of Present Illness:.*?(?=Past Medical History:|Review of Systems:|Physical Exam:)',
            'past_medical': r'Past Medical History:.*?(?=Social History:|Family History:|Physical Exam:)',
            'social_history': r'Social History:.*?(?=Family History:|Physical Exam:)',
            'family_history': r'Family History:.*?(?=Physical Exam:)',
            'physical_exam': r'Physical Exam:.*?(?=Pertinent Results:|Brief Hospital Course:)',
            'pertinent_results': r'Pertinent Results:.*?(?=Brief Hospital Course:)',
            'hospital_course': r'Brief Hospital Course:.*?(?=Medications on Admission:|Discharge Medications:)',
            'medications_admission': r'Medications on Admission:.*?(?=Discharge Medications:|Discharge Disposition:)',
            'discharge_medications': r'Discharge Medications:.*?(?=Discharge Disposition:|Discharge Diagnosis:)',
            'discharge_diagnosis': r'Discharge Diagnosis:.*?(?=Discharge Condition:|Discharge Instructions:)',
            'discharge_condition': r'Discharge Condition:.*?(?=Discharge Instructions:|Followup Instructions:)',
            'discharge_instructions': r'Discharge Instructions:.*?(?=Followup Instructions:|$)',
        }
        
        section_analysis = {}
        for record in self.records[:10]:  # Analyze first 10 records
            note_text = record.get('note_text', '')
            record_sections = {}
            
            for section_name, pattern in section_patterns.items():
                match = re.search(pattern, note_text, re.DOTALL | re.IGNORECASE)
                if match:
                    section_text = match.group(0)
                    record_sections[section_name] = {
                        'length': len(section_text),
                        'present': True
                    }
                else:
                    record_sections[section_name] = {
                        'length': 0,
                        'present': False
                    }
            
            section_analysis[record.get('hadm_id', 'unknown')] = record_sections
        
        return section_analysis
    
    def create_essential_sections_dataset(self, output_path: str = None) -> str:
        """
        Create a dataset with only the most essential sections for medical coding.
        
        Essential sections for ICD/DRG coding:
        - Chief Complaint
        - History of Present Illness
        - Past Medical History
        - Major Procedures
        - Brief Hospital Course (for complications/treatments)
        - Discharge Diagnosis
        """
        essential_patterns = {
            'chief_complaint': r'Chief Complaint:.*?(?=Major Surgical|History of Present)',
            'procedures': r'Major Surgical or Invasive Procedure:.*?(?=History of Present|Past Medical)',
            'history_present': r'History of Present Illness:.*?(?=Past Medical History:|Review of Systems:|Physical Exam:)',
            'past_medical': r'Past Medical History:.*?(?=Social History:|Family History:|Physical Exam:)',
            'hospital_course': r'Brief Hospital Course:.*?(?=Medications on Admission:|Discharge Medications:)',
            'discharge_diagnosis': r'Discharge Diagnosis:.*?(?=Discharge Condition:|Discharge Instructions:)',
        }
        
        optimized_records = []
        total_chars_original = 0
        total_chars_optimized = 0
        
        for record in self.records:
            note_text = record.get('note_text', '')
            total_chars_original += len(note_text)
            
            # Extract essential sections
            essential_sections = []
            for section_name, pattern in essential_patterns.items():
                match = re.search(pattern, note_text, re.DOTALL | re.IGNORECASE)
                if match:
                    section_text = match.group(0).strip()
                    essential_sections.append(f"=== {section_name.upper().replace('_', ' ')} ===")
                    essential_sections.append(section_text)
                    essential_sections.append("")  # Add spacing
            
            # Create optimized note text
            optimized_note = "\n".join(essential_sections)
            total_chars_optimized += len(optimized_note)
            
            # Create new record
            optimized_record = record.copy()
            optimized_record['note_text'] = optimized_note
            optimized_record['optimization_method'] = 'essential_sections'
            optimized_record['original_length'] = len(note_text)
            optimized_record['optimized_length'] = len(optimized_note)
            optimized_record['compression_ratio'] = len(optimized_note) / len(note_text) if len(note_text) > 0 else 0
            
            optimized_records.append(optimized_record)
        
        # Create optimized dataset
        optimized_data = {
            "metadata": {
                **self.data.get("metadata", {}),
                "optimization_method": "essential_sections",
                "optimization_date": pd.Timestamp.now().isoformat(),
                "original_avg_length": total_chars_original / len(self.records),
                "optimized_avg_length": total_chars_optimized / len(optimized_records),
                "compression_ratio": total_chars_optimized / total_chars_original,
                "sections_included": list(essential_patterns.keys())
            },
            "records": optimized_records
        }
        
        # Save optimized dataset
        if output_path is None:
            output_path = self.dataset_path.with_name(
                self.dataset_path.stem + "_optimized_essential_sections.json"
            )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(optimized_data, f, indent=2, ensure_ascii=False)
        
        print(f"Created essential sections dataset: {output_path}")
        print(f"Original avg length: {total_chars_original / len(self.records):.0f} chars")
        print(f"Optimized avg length: {total_chars_optimized / len(optimized_records):.0f} chars")
        print(f"Compression ratio: {total_chars_optimized / total_chars_original:.2f}")
        
        return str(output_path)
    
    def create_summary_based_dataset(self, output_path: str = None, max_length: int = 8000) -> str:
        """
        Create a dataset with AI-generated summaries focused on medical coding.
        This would require LLM calls to generate concise summaries.
        """
        # This is a template - would need LLM integration for actual summarization
        print("Summary-based optimization would require LLM integration.")
        print("This would create concise summaries focused on:")
        print("- Primary diagnoses and complications")
        print("- Major procedures performed")
        print("- Significant comorbidities")
        print("- Key clinical findings")
        return "Not implemented - requires LLM integration"
    
    def create_diagnosis_focused_dataset(self, output_path: str = None) -> str:
        """
        Create a dataset focusing only on diagnostic information and key clinical sections.
        """
        diagnostic_patterns = {
            'chief_complaint': r'Chief Complaint:.*?(?=Major Surgical|History of Present)',
            'procedures': r'Major Surgical or Invasive Procedure:.*?(?=History of Present|Past Medical)',
            'assessment_plan': r'(?:Assessment and Plan|Brief Hospital Course):.*?(?=Medications|Discharge)',
            'discharge_diagnosis': r'Discharge Diagnosis:.*?(?=Discharge Condition:|Discharge Instructions:)',
            'primary_diagnosis': r'(?:Primary|Principal) Diagnosis:.*?(?=Secondary|Discharge Condition)',
            'secondary_diagnosis': r'Secondary Diagnosis:.*?(?=Discharge Condition:|Discharge Instructions:)',
        }
        
        # Also extract any mention of ICD codes in the text
        icd_pattern = r'(?:ICD[- ]?(?:9|10)[- ]?(?:CM)?[- ]?(?:code)?s?:?\s*)([\d\w\.,\s\-]+)'
        
        diagnostic_records = []
        total_chars_original = 0
        total_chars_diagnostic = 0
        
        for record in self.records:
            note_text = record.get('note_text', '')
            total_chars_original += len(note_text)
            
            # Extract diagnostic sections
            diagnostic_sections = []
            
            # Add a concise header with admission info
            header_match = re.search(r'Service:\s*(\w+).*?Chief Complaint:', note_text, re.DOTALL | re.IGNORECASE)
            if header_match:
                diagnostic_sections.append("=== ADMISSION INFO ===")
                diagnostic_sections.append(header_match.group(0))
                diagnostic_sections.append("")
            
            for section_name, pattern in diagnostic_patterns.items():
                match = re.search(pattern, note_text, re.DOTALL | re.IGNORECASE)
                if match:
                    section_text = match.group(0).strip()
                    diagnostic_sections.append(f"=== {section_name.upper().replace('_', ' ')} ===")
                    diagnostic_sections.append(section_text)
                    diagnostic_sections.append("")
            
            # Look for any ICD codes mentioned in the text
            icd_matches = re.findall(icd_pattern, note_text, re.IGNORECASE)
            if icd_matches:
                diagnostic_sections.append("=== ICD CODES MENTIONED IN TEXT ===")
                for match in icd_matches:
                    diagnostic_sections.append(f"ICD Codes: {match.strip()}")
                diagnostic_sections.append("")
            
            # Create diagnostic-focused note text
            diagnostic_note = "\n".join(diagnostic_sections)
            total_chars_diagnostic += len(diagnostic_note)
            
            # Create new record
            diagnostic_record = record.copy()
            diagnostic_record['note_text'] = diagnostic_note
            diagnostic_record['optimization_method'] = 'diagnosis_focused'
            diagnostic_record['original_length'] = len(note_text)
            diagnostic_record['optimized_length'] = len(diagnostic_note)
            diagnostic_record['compression_ratio'] = len(diagnostic_note) / len(note_text) if len(note_text) > 0 else 0
            
            diagnostic_records.append(diagnostic_record)
        
        # Create diagnostic dataset
        diagnostic_data = {
            "metadata": {
                **self.data.get("metadata", {}),
                "optimization_method": "diagnosis_focused",
                "optimization_date": pd.Timestamp.now().isoformat(),
                "original_avg_length": total_chars_original / len(self.records),
                "optimized_avg_length": total_chars_diagnostic / len(diagnostic_records),
                "compression_ratio": total_chars_diagnostic / total_chars_original,
                "sections_included": list(diagnostic_patterns.keys())
            },
            "records": diagnostic_records
        }
        
        # Save diagnostic dataset
        if output_path is None:
            output_path = self.dataset_path.with_name(
                self.dataset_path.stem + "_optimized_diagnosis_focused.json"
            )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(diagnostic_data, f, indent=2, ensure_ascii=False)
        
        print(f"Created diagnosis-focused dataset: {output_path}")
        print(f"Original avg length: {total_chars_original / len(self.records):.0f} chars")
        print(f"Optimized avg length: {total_chars_diagnostic / len(diagnostic_records):.0f} chars")
        print(f"Compression ratio: {total_chars_diagnostic / total_chars_original:.2f}")
        
        return str(output_path)
    
    def create_token_length_based_dataset(self, output_path: str = None, max_tokens: int = 4000) -> str:
        """
        Create a dataset that truncates notes to a specific token limit while preserving key sections.
        Prioritizes the most important sections for medical coding.
        """
        # Rough estimation: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        
        # Priority order for sections (most important first)
        priority_patterns = [
            ('discharge_diagnosis', r'Discharge Diagnosis:.*?(?=Discharge Condition:|Discharge Instructions:)'),
            ('chief_complaint', r'Chief Complaint:.*?(?=Major Surgical|History of Present)'),
            ('procedures', r'Major Surgical or Invasive Procedure:.*?(?=History of Present|Past Medical)'),
            ('history_present', r'History of Present Illness:.*?(?=Past Medical History:|Review of Systems:|Physical Exam:)'),
            ('past_medical', r'Past Medical History:.*?(?=Social History:|Family History:|Physical Exam:)'),
            ('hospital_course', r'Brief Hospital Course:.*?(?=Medications on Admission:|Discharge Medications:)'),
            ('physical_exam', r'Physical Exam:.*?(?=Pertinent Results:|Brief Hospital Course:)'),
            ('pertinent_results', r'Pertinent Results:.*?(?=Brief Hospital Course:)'),
        ]
        
        truncated_records = []
        total_chars_original = 0
        total_chars_truncated = 0
        
        for record in self.records:
            note_text = record.get('note_text', '')
            total_chars_original += len(note_text)
            
            # Extract sections in priority order
            truncated_sections = []
            current_length = 0
            
            for section_name, pattern in priority_patterns:
                if current_length >= max_chars:
                    break
                    
                match = re.search(pattern, note_text, re.DOTALL | re.IGNORECASE)
                if match:
                    section_text = match.group(0).strip()
                    
                    # Check if adding this section would exceed limit
                    section_header = f"=== {section_name.upper().replace('_', ' ')} ==="
                    section_with_header = f"{section_header}\n{section_text}\n"
                    
                    if current_length + len(section_with_header) <= max_chars:
                        truncated_sections.append(section_with_header)
                        current_length += len(section_with_header)
                    else:
                        # Truncate this section to fit
                        remaining_chars = max_chars - current_length - len(section_header) - 2  # 2 for newlines
                        if remaining_chars > 100:  # Only include if we can fit meaningful content
                            truncated_text = section_text[:remaining_chars] + "... [TRUNCATED]"
                            truncated_sections.append(f"{section_header}\n{truncated_text}\n")
                        break
            
            # Create truncated note text
            truncated_note = "".join(truncated_sections)
            total_chars_truncated += len(truncated_note)
            
            # Create new record
            truncated_record = record.copy()
            truncated_record['note_text'] = truncated_note
            truncated_record['optimization_method'] = 'token_length_based'
            truncated_record['max_tokens'] = max_tokens
            truncated_record['original_length'] = len(note_text)
            truncated_record['optimized_length'] = len(truncated_note)
            truncated_record['compression_ratio'] = len(truncated_note) / len(note_text) if len(note_text) > 0 else 0
            
            truncated_records.append(truncated_record)
        
        # Create truncated dataset
        truncated_data = {
            "metadata": {
                **self.data.get("metadata", {}),
                "optimization_method": "token_length_based",
                "max_tokens": max_tokens,
                "max_chars_estimate": max_chars,
                "optimization_date": pd.Timestamp.now().isoformat(),
                "original_avg_length": total_chars_original / len(self.records),
                "optimized_avg_length": total_chars_truncated / len(truncated_records),
                "compression_ratio": total_chars_truncated / total_chars_original,
                "priority_sections": [p[0] for p in priority_patterns]
            },
            "records": truncated_records
        }
        
        # Save truncated dataset
        if output_path is None:
            output_path = self.dataset_path.with_name(
                self.dataset_path.stem + f"_optimized_max_{max_tokens}_tokens.json"
            )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(truncated_data, f, indent=2, ensure_ascii=False)
        
        print(f"Created token-limited dataset: {output_path}")
        print(f"Max tokens: {max_tokens} (~{max_chars} chars)")
        print(f"Original avg length: {total_chars_original / len(self.records):.0f} chars")
        print(f"Optimized avg length: {total_chars_truncated / len(truncated_records):.0f} chars")
        print(f"Compression ratio: {total_chars_truncated / total_chars_original:.2f}")
        
        return str(output_path)
    
    def generate_optimization_report(self) -> str:
        """Generate a comprehensive report on optimization opportunities."""
        
        # Analyze current dataset
        note_lengths = [len(record.get('note_text', '')) for record in self.records]
        
        report = []
        report.append("=== MIMIC Dataset Optimization Report ===\n")
        
        # Current stats
        report.append("CURRENT DATASET STATISTICS:")
        report.append(f"  Total records: {len(self.records)}")
        report.append(f"  Average note length: {statistics.mean(note_lengths):.0f} characters")
        report.append(f"  Median note length: {statistics.median(note_lengths):.0f} characters")
        report.append(f"  Min/Max length: {min(note_lengths):.0f} / {max(note_lengths):.0f} characters")
        report.append(f"  Standard deviation: {statistics.stdev(note_lengths):.0f} characters\n")
        
        # Token estimates
        avg_tokens = statistics.mean(note_lengths) / 4  # Rough estimate
        report.append("TOKEN ESTIMATES (rough):")
        report.append(f"  Average tokens per note: {avg_tokens:.0f}")
        report.append(f"  Cost estimate per note (GPT-4): ${avg_tokens * 0.03 / 1000:.3f}")
        report.append(f"  Total cost for full dataset: ${len(self.records) * avg_tokens * 0.03 / 1000:.2f}\n")
        
        # Optimization opportunities
        report.append("OPTIMIZATION STRATEGIES AVAILABLE:")
        report.append("  1. Essential Sections Only:")
        report.append("     - Keep: Chief Complaint, History, Procedures, Hospital Course, Diagnosis")
        report.append("     - Remove: Demographics, Social History, detailed lab results")
        report.append("     - Expected compression: ~40-60%\n")
        
        report.append("  2. Diagnosis-Focused:")
        report.append("     - Keep: All diagnostic information and assessment sections")
        report.append("     - Remove: Procedural details, medications, instructions")
        report.append("     - Expected compression: ~50-70%\n")
        
        report.append("  3. Token-Length Limited:")
        report.append("     - Truncate to specific token limits (e.g., 2000, 4000, 6000 tokens)")
        report.append("     - Preserve most important sections first")
        report.append("     - Configurable compression based on token limit\n")
        
        report.append("RECOMMENDATIONS:")
        report.append("  - For initial experimentation: Use Essential Sections optimization")
        report.append("  - For production: Use Token-Length Limited with 4000 token limit")
        report.append("  - For research: Compare performance across different optimization strategies")
        report.append("  - Always validate that essential medical information is preserved")
        
        return "\n".join(report)


def main():
    """Main function to demonstrate optimization strategies."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize MIMIC dataset for efficient processing")
    parser.add_argument(
        "--dataset", 
        default="/home/justjosh/Turing-Test/src/data/mimic/mimic_iv_medical_coding_evaluation_dataset.json",
        help="Path to MIMIC dataset JSON file"
    )
    parser.add_argument(
        "--strategy", 
        choices=["essential", "diagnosis", "tokens", "report"],
        default="report",
        help="Optimization strategy to apply"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
        help="Maximum tokens for token-based optimization"
    )
    
    args = parser.parse_args()
    
    optimizer = MIMICDatasetOptimizer(args.dataset)
    
    if args.strategy == "essential":
        output_path = optimizer.create_essential_sections_dataset()
        print(f"Created optimized dataset: {output_path}")
        
    elif args.strategy == "diagnosis":
        output_path = optimizer.create_diagnosis_focused_dataset()
        print(f"Created optimized dataset: {output_path}")
        
    elif args.strategy == "tokens":
        output_path = optimizer.create_token_length_based_dataset(max_tokens=args.max_tokens)
        print(f"Created optimized dataset: {output_path}")
        
    elif args.strategy == "report":
        report = optimizer.generate_optimization_report()
        print(report)
        
        # Also analyze note structure
        print("\n=== ANALYZING NOTE STRUCTURE ===")
        section_analysis = optimizer.analyze_note_structure()
        print("Section analysis completed for first 10 records.")
        
        # Show which sections are most common
        section_presence = {}
        for record_analysis in section_analysis.values():
            for section, info in record_analysis.items():
                if info['present']:
                    section_presence[section] = section_presence.get(section, 0) + 1
        
        print("\nSection presence across analyzed records:")
        for section, count in sorted(section_presence.items(), key=lambda x: x[1], reverse=True):
            print(f"  {section}: {count}/10 records ({count*10}%)")


if __name__ == "__main__":
    main()
