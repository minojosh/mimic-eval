"""
Enhanced MIMIC-IV Medical Coding Dataset Creator
Creates structured "findings → codes" format for efficient agentic RAG retrieval.

This script transforms raw MIMIC-IV data into the exact format requested:
- Primary vs Secondary findings (based on seq_num)
- Human-readable descriptions + ICD/DRG codes
- Structured JSON per hospital admission (hadm_id)
- Optimized for LLM reasoning and RAG retrieval

Output format:
{
  "hadm_id": 123456,
  "Primary findings": [
    { "finding": "Headache, unspecified", "code": "ICD-10 R51.9" }
  ],
  "Secondary findings": [
    { "finding": "Hypertension, essential (primary)", "code": "ICD-10 I10" }
  ],
  "Procedures": [
    { "procedure": "CT scan of brain", "code": "ICD-10-PCS 4A00X4Z" }
  ],
  "DRG": [
    { "drg_code": "101", "description": "Seizures w/o MCC" }
  ]
}
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StructuredMedicalCodingProcessor:
    """
    Processes MIMIC-IV data into structured findings → codes format
    optimized for agentic RAG systems.
    """
    
    def __init__(self, data_dir: str = "/home/justjosh/Turing-Test/mimic-eval/data"):
        """Initialize the processor with data directory."""
        self.data_dir = Path(data_dir)
        self.hosp_dir = self.data_dir / "hosp"
        self.note_dir = self.data_dir / "note"
        
        # Data containers
        self.diagnoses_df = None
        self.procedures_df = None
        self.drg_df = None
        self.discharge_notes_df = None
        self.icd_diag_descriptions = None
        self.icd_proc_descriptions = None
        
        logger.info(f"Initialized processor with data directory: {self.data_dir}")
    
    def load_all_data(self) -> bool:
        """Load all required MIMIC-IV data files."""
        try:
            logger.info("Loading MIMIC-IV data files...")
            
            # Load ICD diagnoses with sequence numbers
            diag_path = self.hosp_dir / "diagnoses_icd.csv"
            if diag_path.exists():
                self.diagnoses_df = pd.read_csv(diag_path)
                logger.info(f"✅ Loaded {len(self.diagnoses_df)} ICD diagnosis records")
            
            # Load ICD procedures with sequence numbers  
            proc_path = self.hosp_dir / "procedures_icd.csv"
            if proc_path.exists():
                self.procedures_df = pd.read_csv(proc_path)
                logger.info(f"✅ Loaded {len(self.procedures_df)} ICD procedure records")
            
            # Load DRG codes
            drg_path = self.hosp_dir / "drgcodes.csv"
            if drg_path.exists():
                self.drg_df = pd.read_csv(drg_path)
                logger.info(f"✅ Loaded {len(self.drg_df)} DRG code records")
            
            # Load discharge notes
            notes_path = self.note_dir / "discharge.csv"
            if notes_path.exists():
                self.discharge_notes_df = pd.read_csv(notes_path)
                logger.info(f"✅ Loaded {len(self.discharge_notes_df)} discharge notes")
            
            # Load ICD diagnosis descriptions
            diag_desc_path = self.hosp_dir / "d_icd_diagnoses.csv"
            if diag_desc_path.exists():
                self.icd_diag_descriptions = pd.read_csv(diag_desc_path)
                logger.info(f"✅ Loaded {len(self.icd_diag_descriptions)} ICD diagnosis descriptions")
            
            # Load ICD procedure descriptions
            proc_desc_path = self.hosp_dir / "d_icd_procedures.csv"
            if proc_desc_path.exists():
                self.icd_proc_descriptions = pd.read_csv(proc_desc_path)
                logger.info(f"✅ Loaded {len(self.icd_proc_descriptions)} ICD procedure descriptions")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def get_icd_description(self, icd_code: str, icd_version: int, is_procedure: bool = False) -> str:
        """Get human-readable description for an ICD code."""
        try:
            # Choose the appropriate description table
            desc_df = self.icd_proc_descriptions if is_procedure else self.icd_diag_descriptions
            
            if desc_df is None:
                return f"Unknown {'procedure' if is_procedure else 'condition'}"
            
            # Find matching description
            match = desc_df[
                (desc_df['icd_code'] == icd_code) & 
                (desc_df['icd_version'] == icd_version)
            ]
            
            if not match.empty:
                return match.iloc[0]['long_title']
            else:
                return f"Unknown {'procedure' if is_procedure else 'condition'} (Code: {icd_code})"
                
        except Exception as e:
            logger.warning(f"Error getting description for {icd_code}: {e}")
            return f"Code: {icd_code}"
    
    def format_icd_code(self, icd_code: str, icd_version: int) -> str:
        """Format ICD code with proper version prefix."""
        if icd_version == 9:
            return f"ICD-9 {icd_code}"
        elif icd_version == 10:
            return f"ICD-10 {icd_code}"
        else:
            return f"ICD-{icd_version} {icd_code}"
    
    def process_admission_codes(self, hadm_id: int) -> Dict[str, Any]:
        """
        Process all codes for a single hospital admission into structured format.
        Maintains compatibility with existing mimic_iv_note format while adding structured findings.
        
        Args:
            hadm_id: Hospital admission ID
            
        Returns:
            Structured dictionary with enhanced findings format + original fields
        """
        # Initialize with base structure (compatible with existing format)
        result = {
            "hadm_id": int(hadm_id),  # Ensure native Python int
            "subject_id": None,
            "note_text": "",
            "icd_codes": "",  # Original comma-separated format for backward compatibility
            "icd_versions": "",  # Original comma-separated format
            "primary_icd_version": None,
            "drg_codes": "",  # Original comma-separated format
            "note_id": "",
            "chartdate": "",
            "storetime": "",
            "note_type": "discharge_summary",
            # NEW: Enhanced structured format
            "structured_findings": {
                "Primary findings": [],
                "Secondary findings": [],
                "Procedures": [],
                "DRG": []
            }
        }
        
        # Lists to build original comma-separated format
        all_icd_codes = []
        all_icd_versions = []
        
        try:
            # Process ICD diagnoses (primary vs secondary based on seq_num)
            if self.diagnoses_df is not None:
                admission_diagnoses = self.diagnoses_df[
                    self.diagnoses_df['hadm_id'] == hadm_id
                ].sort_values('seq_num')
                
                for _, row in admission_diagnoses.iterrows():
                    icd_code = str(row['icd_code'])
                    icd_version = int(row['icd_version'])
                    seq_num = int(row['seq_num'])
                    
                    # Add to original format lists
                    all_icd_codes.append(icd_code)
                    all_icd_versions.append(str(icd_version))
                    
                    # Get human-readable description
                    description = self.get_icd_description(icd_code, icd_version, is_procedure=False)
                    formatted_code = self.format_icd_code(icd_code, icd_version)
                    
                    finding_entry = {
                        "finding": description,
                        "code": formatted_code,
                        "sequence": int(seq_num)  # Ensure native Python int
                    }
                    
                    # Primary finding (seq_num = 1) vs Secondary findings (seq_num > 1)
                    if seq_num == 1:
                        result["structured_findings"]["Primary findings"].append(finding_entry)
                        # Set primary ICD version from the first primary diagnosis
                        if result["primary_icd_version"] is None:
                            result["primary_icd_version"] = int(icd_version)  # Ensure native Python int
                    else:
                        result["structured_findings"]["Secondary findings"].append(finding_entry)
            
            # Process ICD procedures
            if self.procedures_df is not None:
                admission_procedures = self.procedures_df[
                    self.procedures_df['hadm_id'] == hadm_id
                ].sort_values('seq_num')
                
                for _, row in admission_procedures.iterrows():
                    icd_code = str(row['icd_code'])
                    icd_version = int(row['icd_version'])
                    seq_num = int(row['seq_num'])
                    
                    # Add to original format lists (procedures are separate from diagnoses)
                    # Note: procedures are typically tracked separately in MIMIC-IV
                    
                    # Get human-readable description
                    description = self.get_icd_description(icd_code, icd_version, is_procedure=True)
                    formatted_code = self.format_icd_code(icd_code, icd_version)
                    
                    procedure_entry = {
                        "procedure": description,
                        "code": formatted_code,
                        "sequence": int(seq_num),  # Ensure native Python int
                        "chartdate": str(row.get('chartdate', '')) if pd.notna(row.get('chartdate')) else ""
                    }
                    
                    result["structured_findings"]["Procedures"].append(procedure_entry)
            
            # Process DRG codes
            drg_code_list = []
            if self.drg_df is not None:
                admission_drg = self.drg_df[self.drg_df['hadm_id'] == hadm_id]
                
                for _, row in admission_drg.iterrows():
                    drg_code = str(row['drg_code'])
                    description = str(row['description']) if pd.notna(row['description']) else f"DRG {drg_code}"
                    
                    # Add to original format list
                    drg_code_list.append(drg_code)
                    
                    drg_entry = {
                        "drg_code": drg_code,
                        "description": description
                    }
                    
                    # Add optional severity and mortality if available
                    if pd.notna(row.get('drg_severity')):
                        drg_entry["severity"] = str(row['drg_severity'])
                    if pd.notna(row.get('drg_mortality')):
                        drg_entry["mortality"] = str(row['drg_mortality'])
                    if pd.notna(row.get('drg_type')):
                        drg_entry["drg_type"] = str(row['drg_type'])
                    
                    result["structured_findings"]["DRG"].append(drg_entry)
            
            # Set original format fields for backward compatibility
            result["icd_codes"] = ",".join(all_icd_codes)
            result["icd_versions"] = ",".join(all_icd_versions)
            result["drg_codes"] = ",".join(drg_code_list)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing admission {hadm_id}: {e}")
            return result
    
    def get_overlapping_admissions(self) -> List[int]:
        """Find hospital admissions that have notes AND codes."""
        try:
            # Get hadm_ids that have data in each table
            note_hadms = set()
            if self.discharge_notes_df is not None:
                note_hadms = set(self.discharge_notes_df['hadm_id'].dropna())
            
            diag_hadms = set()
            if self.diagnoses_df is not None:
                diag_hadms = set(self.diagnoses_df['hadm_id'].dropna())
            
            proc_hadms = set()
            if self.procedures_df is not None:
                proc_hadms = set(self.procedures_df['hadm_id'].dropna())
            
            drg_hadms = set()
            if self.drg_df is not None:
                drg_hadms = set(self.drg_df['hadm_id'].dropna())
            
            # Find admissions with at least notes and diagnoses
            overlap_hadms = note_hadms.intersection(diag_hadms)
            
            logger.info(f"Data overlap analysis:")
            logger.info(f"  Discharge notes: {len(note_hadms)} admissions")
            logger.info(f"  ICD diagnoses: {len(diag_hadms)} admissions") 
            logger.info(f"  ICD procedures: {len(proc_hadms)} admissions")
            logger.info(f"  DRG codes: {len(drg_hadms)} admissions")
            logger.info(f"  Notes + Diagnoses overlap: {len(overlap_hadms)} admissions")
            
            return sorted(list(overlap_hadms))
            
        except Exception as e:
            logger.error(f"Error finding overlapping admissions: {e}")
            return []
    
    def create_structured_dataset(self, sample_size: Optional[int] = None, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Create the complete structured dataset with findings → codes format.
        
        Args:
            sample_size: Number of admissions to process (None for all)
            output_file: Output JSON file path (optional)
            
        Returns:
            Structured dataset dictionary
        """
        logger.info("Creating structured medical coding dataset...")
        
        # Get overlapping admissions
        overlap_hadms = self.get_overlapping_admissions()
        
        if not overlap_hadms:
            raise ValueError("No overlapping admissions found between notes and codes!")
        
        # Sample if requested
        if sample_size and sample_size < len(overlap_hadms):
            import random
            random.seed(42)  # For reproducibility
            overlap_hadms = random.sample(overlap_hadms, sample_size)
            logger.info(f"Sampling {sample_size} admissions from {len(overlap_hadms)} available")
        
        # Process each admission
        structured_records = []
        for i, hadm_id in enumerate(overlap_hadms):
            if i % 50 == 0:
                logger.info(f"Processing admission {i+1}/{len(overlap_hadms)}: {hadm_id}")
            
            # Get structured codes (includes both original format and new structured format)
            admission_data = self.process_admission_codes(hadm_id)
            
            # Add note text and metadata if available
            if self.discharge_notes_df is not None:
                note_row = self.discharge_notes_df[
                    self.discharge_notes_df['hadm_id'] == hadm_id
                ]
                if not note_row.empty:
                    note_data = note_row.iloc[0]
                    admission_data["note_text"] = str(note_data.get('text', ''))
                    admission_data["subject_id"] = int(note_data.get('subject_id', 0))  # Ensure native Python int
                    admission_data["note_id"] = str(note_data.get('note_id', ''))
                    admission_data["chartdate"] = str(note_data.get('chartdate', ''))
                    admission_data["storetime"] = str(note_data.get('storetime', ''))
            
            structured_records.append(admission_data)
        
        # Create final dataset with enhanced metadata
        dataset = {
            "metadata": {
                "dataset_name": "MIMIC-IV Enhanced Structured Medical Coding Dataset",
                "creation_date": datetime.now().isoformat(),
                "total_admissions": len(structured_records),
                "format": "enhanced_mimic_iv_note_format",
                "description": "Enhanced dataset maintaining original mimic_iv_note format with added structured findings → codes mapping",
                "enhancements": [
                    "structured_findings with Primary/Secondary findings separation",
                    "human_readable_descriptions for all ICD codes",
                    "sequence_number_based_ordering",
                    "enhanced_DRG_information",
                    "backward_compatibility_with_original_format"
                ],
                "data_sources": {
                    "diagnoses": len(self.diagnoses_df) if self.diagnoses_df is not None else 0,
                    "procedures": len(self.procedures_df) if self.procedures_df is not None else 0,
                    "drg_codes": len(self.drg_df) if self.drg_df is not None else 0,
                    "discharge_notes": len(self.discharge_notes_df) if self.discharge_notes_df is not None else 0,
                    "icd_diagnosis_descriptions": len(self.icd_diag_descriptions) if self.icd_diag_descriptions is not None else 0,
                    "icd_procedure_descriptions": len(self.icd_proc_descriptions) if self.icd_proc_descriptions is not None else 0
                },
                "compatibility": "Maintains all original fields from mimic_iv_note_essential_sections.json"
            },
            "records": structured_records
        }
        
        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Structured dataset saved to: {output_path}")
        
        # Print summary statistics
        self._print_dataset_summary(dataset)
        
        return dataset
    
    def _print_dataset_summary(self, dataset: Dict[str, Any]):
        """Print summary statistics for the created dataset."""
        records = dataset["records"]
        
        logger.info("\n" + "="*60)
        logger.info("ENHANCED STRUCTURED DATASET SUMMARY")
        logger.info("="*60)
        
        logger.info(f"📊 Total admissions: {len(records)}")
        
        # Count findings from structured_findings
        total_primary = sum(len(r["structured_findings"]["Primary findings"]) for r in records)
        total_secondary = sum(len(r["structured_findings"]["Secondary findings"]) for r in records)
        total_procedures = sum(len(r["structured_findings"]["Procedures"]) for r in records)
        total_drg = sum(len(r["structured_findings"]["DRG"]) for r in records)
        
        logger.info(f"🏥 Primary findings: {total_primary}")
        logger.info(f"📋 Secondary findings: {total_secondary}")
        logger.info(f"⚕️  Procedures: {total_procedures}")
        logger.info(f"💊 DRG codes: {total_drg}")
        
        # Count records with note text
        records_with_notes = sum(1 for r in records if r.get("note_text", "").strip())
        logger.info(f"📝 Records with note text: {records_with_notes}/{len(records)}")
        
        # Averages per admission
        avg_primary = total_primary / len(records) if records else 0
        avg_secondary = total_secondary / len(records) if records else 0
        avg_procedures = total_procedures / len(records) if records else 0
        avg_drg = total_drg / len(records) if records else 0
        
        logger.info(f"\n📈 Averages per admission:")
        logger.info(f"   Primary findings: {avg_primary:.1f}")
        logger.info(f"   Secondary findings: {avg_secondary:.1f}")
        logger.info(f"   Procedures: {avg_procedures:.1f}")
        logger.info(f"   DRG codes: {avg_drg:.1f}")
        
        # Show sample record structure
        if records:
            sample = records[0]
            logger.info(f"\n🔍 Sample record structure (HADM {sample['hadm_id']}):")
            logger.info(f"   📝 Note text length: {len(sample.get('note_text', ''))} characters")
            logger.info(f"   🏷️  Original ICD codes: {sample.get('icd_codes', '')[:50]}...")
            logger.info(f"   📊 Original DRG codes: {sample.get('drg_codes', '')}")
            
            # Show structured findings sample
            structured = sample["structured_findings"]
            if structured["Primary findings"]:
                primary = structured["Primary findings"][0]
                logger.info(f"   🎯 Primary: {primary['finding'][:50]}... → {primary['code']}")
            
            if structured["Secondary findings"]:
                secondary = structured["Secondary findings"][0]
                logger.info(f"   📋 Secondary: {secondary['finding'][:50]}... → {secondary['code']}")
            
            if structured["Procedures"]:
                procedure = structured["Procedures"][0]
                logger.info(f"   ⚕️  Procedure: {procedure['procedure'][:50]}... → {procedure['code']}")
            
            if structured["DRG"]:
                drg = structured["DRG"][0]
                logger.info(f"   💊 DRG: {drg['description'][:50]}... → {drg['drg_code']}")
        
        logger.info(f"\n✅ BACKWARD COMPATIBILITY:")
        logger.info(f"   🔄 Maintains all original mimic_iv_note fields")
        logger.info(f"   ➕ Adds structured_findings for enhanced LLM reasoning")
        logger.info(f"   🎯 Ready for both existing and new RAG workflows")


def main():
    """Main execution function."""
    
    # Initialize processor
    processor = StructuredMedicalCodingProcessor()
    
    # Load all data
    if not processor.load_all_data():
        logger.error("Failed to load required data files!")
        return
    
    # Create structured dataset
    output_file = "/home/justjosh/Turing-Test/mimic-eval/data/enhanced_mimic_iv_structured_coding.json"
    
    try:
        dataset = processor.create_structured_dataset(
            sample_size=None,  # Process ALL admissions with required data (notes + codes)
            output_file=output_file
        )
        
        logger.info(f"\n✅ SUCCESS: Enhanced structured dataset created!")
        logger.info(f"📁 Output file: {output_file}")
        logger.info(f"🔄 Backward compatible with existing mimic_iv_note format")
        logger.info(f"➕ Enhanced with structured findings → codes mapping")
        logger.info(f"🎯 Ready for agentic RAG integration!")
        
        # Show format comparison
        if dataset["records"]:
            sample = dataset["records"][0]
            logger.info(f"\n📋 SAMPLE ENHANCED RECORD FORMAT:")
            logger.info(f"   hadm_id: {sample['hadm_id']}")
            logger.info(f"   subject_id: {sample['subject_id']}")
            logger.info(f"   note_text: {len(sample.get('note_text', ''))} chars")
            logger.info(f"   icd_codes (original): {sample['icd_codes'][:50]}...")
            logger.info(f"   structured_findings: {len(sample['structured_findings']['Primary findings'])} primary, {len(sample['structured_findings']['Secondary findings'])} secondary")
            
            if sample['structured_findings']['Primary findings']:
                primary_sample = sample['structured_findings']['Primary findings'][0]
                logger.info(f"   📋 Primary example: \"{primary_sample['finding'][:40]}...\" → {primary_sample['code']}")
        
    except Exception as e:
        logger.error(f"❌ Error creating dataset: {e}")
        raise


if __name__ == "__main__":
    main()
