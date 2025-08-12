"""
Creates a proper medical coding evaluation dataset by joining MIMIC-IV-Note discharge summaries 
with MIMIC-IV hospital ICD and DRG codes.

This script:
1. Loads discharge summaries from MIMIC-IV-Note
2. Loads ICD diagnoses (with version info) and DRG codes from MIMIC-IV hospital data  
3. Joins them on hadm_id to create samples with both clinical notes and ground truth codes
4. Preserves ICD version information (ICD-9 vs ICD-10) for version-appropriate evaluation
5. Samples N records for LLM evaluation
6. Outputs both CSV and JSON formats ready for medical coding evaluation
"""
import pandas as pd
import random
import json
from pathlib import Path

# Configuration
SAMPLE_SIZE = 1000  # Number of samples for evaluation
RANDOM_SEED = 42  # For reproducible sampling

# Paths to MIMIC-IV data
DATA_DIR = Path("/home/justjosh/Turing-Test/src/data/mimic")
DISCHARGE_NOTES_PATH = DATA_DIR / "note" / "discharge.csv"
ICD_DIAGNOSES_PATH = DATA_DIR / "hosp" / "diagnoses_icd.csv" 
DRG_CODES_PATH = DATA_DIR / "hosp" / "drgcodes.csv"
OUTPUT_CSV_PATH = DATA_DIR / "mimic_iv_medical_coding_evaluation_dataset.csv"
OUTPUT_JSON_PATH = DATA_DIR / "mimic_iv_medical_coding_evaluation_dataset.json"

def load_and_sample_data():
    """Load and process MIMIC-IV data to create evaluation dataset."""
    
    print("Loading MIMIC-IV discharge summaries...")
    # Load discharge summaries
    discharge_notes = pd.read_csv(DISCHARGE_NOTES_PATH)
    print(f"Loaded {len(discharge_notes)} discharge summaries")
    print(f"Discharge summary columns: {discharge_notes.columns.tolist()}")
    
    print("\nLoading MIMIC-IV ICD diagnoses...")
    # Load ICD diagnoses 
    diagnoses = pd.read_csv(ICD_DIAGNOSES_PATH)
    print(f"Loaded {len(diagnoses)} ICD diagnosis records")
    print(f"ICD diagnoses columns: {diagnoses.columns.tolist()}")
    
    print("\nLoading MIMIC-IV DRG codes...")
    # Load DRG codes
    drg_codes = pd.read_csv(DRG_CODES_PATH) 
    print(f"Loaded {len(drg_codes)} DRG code records")
    print(f"DRG codes columns: {drg_codes.columns.tolist()}")
    
    # Check overlaps
    note_hadms = set(discharge_notes['hadm_id'].dropna())
    icd_hadms = set(diagnoses['hadm_id'].dropna()) 
    drg_hadms = set(drg_codes['hadm_id'].dropna())
    
    print(f"\nData overlap analysis:")
    print(f"Discharge notes HADMs: {len(note_hadms)}")
    print(f"ICD diagnoses HADMs: {len(icd_hadms)}")
    print(f"DRG codes HADMs: {len(drg_hadms)}")
    
    # Find HADMs that have all three: notes, ICD codes, and DRG codes
    complete_hadms = note_hadms.intersection(icd_hadms).intersection(drg_hadms)
    print(f"HADMs with notes + ICD + DRG: {len(complete_hadms)}")
    
    if len(complete_hadms) == 0:
        print("WARNING: No HADMs found with all three data types!")
        # Fall back to notes + ICD only
        complete_hadms = note_hadms.intersection(icd_hadms)
        print(f"Falling back to HADMs with notes + ICD only: {len(complete_hadms)}")
        
        if len(complete_hadms) == 0:
            raise ValueError("No overlapping HADMs found between notes and diagnoses!")
    
    # Sample HADMs for evaluation
    sample_size = min(SAMPLE_SIZE, len(complete_hadms))
    print(f"\nSampling {sample_size} HADMs for evaluation...")
    
    random.seed(RANDOM_SEED)
    sampled_hadms = random.sample(list(complete_hadms), sample_size)
    print(f"Selected HADMs: {sampled_hadms[:5]}..." if len(sampled_hadms) > 5 else f"Selected HADMs: {sampled_hadms}")
    
    return discharge_notes, diagnoses, drg_codes, sampled_hadms

def create_evaluation_dataset(discharge_notes, diagnoses, drg_codes, sampled_hadms):
    """Create the final evaluation dataset with ICD version information."""
    
    print("\nCreating evaluation dataset...")
    
    # Group ICD codes and versions by HADM_ID
    icd_data = diagnoses.groupby('hadm_id').agg({
        'icd_code': list,
        'icd_version': list
    }).to_dict('index')
    
    # Group DRG codes by HADM_ID  
    drg_grouped = drg_codes.groupby('hadm_id')['drg_code'].apply(list).to_dict()
    
    # Create evaluation records
    evaluation_records = []
    
    for hadm_id in sampled_hadms:
        # Get discharge summary
        note_row = discharge_notes[discharge_notes['hadm_id'] == hadm_id].iloc[0]
        
        # Get ICD codes and versions
        icd_info = icd_data.get(hadm_id, {'icd_code': [], 'icd_version': []})
        icd_codes = icd_info['icd_code']
        icd_versions = icd_info['icd_version']
        
        # Determine the primary ICD version for this admission
        if icd_versions:
            # Use the most common version for this admission
            version_counts = pd.Series(icd_versions).value_counts()
            primary_icd_version = version_counts.index[0]
        else:
            primary_icd_version = None
        
        # Get DRG codes  
        drg_codes_list = drg_grouped.get(hadm_id, [])
        
        # Create evaluation record
        record = {
            'hadm_id': hadm_id,
            'subject_id': note_row.get('subject_id', ''),
            'note_text': note_row.get('text', ''),
            'icd_codes': ','.join(map(str, icd_codes)) if icd_codes else '',
            'icd_versions': ','.join(map(str, icd_versions)) if icd_versions else '',
            'primary_icd_version': primary_icd_version,
            'drg_codes': ','.join(map(str, drg_codes_list)) if drg_codes_list else '',
            'note_id': note_row.get('note_id', ''),
            'chartdate': note_row.get('chartdate', ''),
            'storetime': note_row.get('storetime', ''),
            'note_type': 'discharge_summary'
        }
        
        evaluation_records.append(record)
        
        # Print sample info
        version_info = f"ICD-{primary_icd_version}" if primary_icd_version else "No ICD"
        print(f"HADM {hadm_id}: {len(icd_codes)} ICD codes ({version_info}), {len(drg_codes_list)} DRG codes, {len(record['note_text'])} chars")
    
    return pd.DataFrame(evaluation_records)

def main():
    """Main execution function."""
    
    print("=== MIMIC-IV Medical Coding Evaluation Dataset Creation ===")
    print(f"Target sample size: {SAMPLE_SIZE}")
    print(f"Random seed: {RANDOM_SEED}")
    print()
    
    # Load and sample data
    discharge_notes, diagnoses, drg_codes, sampled_hadms = load_and_sample_data()
    
    # Create evaluation dataset
    evaluation_df = create_evaluation_dataset(discharge_notes, diagnoses, drg_codes, sampled_hadms)
    
    # Save to CSV
    print(f"\nSaving evaluation dataset to: {OUTPUT_CSV_PATH}")
    evaluation_df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    # Save to JSON
    print(f"Saving evaluation dataset to: {OUTPUT_JSON_PATH}")
    # Convert DataFrame to list of dictionaries for JSON
    json_data = evaluation_df.to_dict('records')
    
    # Create structured JSON with metadata
    json_output = {
        "metadata": {
            "dataset_name": "MIMIC-IV Medical Coding Evaluation Dataset",
            "creation_date": pd.Timestamp.now().isoformat(),
            "sample_size": len(evaluation_df),
            "random_seed": RANDOM_SEED,
            "description": "Evaluation dataset for LLM medical coding tasks with ICD and DRG codes"
        },
        "records": json_data
    }
    
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    # Print summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Total records: {len(evaluation_df)}")
    print(f"Records with ICD codes: {(evaluation_df['icd_codes'] != '').sum()}")
    print(f"Records with DRG codes: {(evaluation_df['drg_codes'] != '').sum()}")
    print(f"Average note length: {evaluation_df['note_text'].str.len().mean():.0f} characters")
    print(f"Average ICD codes per record: {evaluation_df['icd_codes'].str.split(',').str.len().mean():.1f}")
    print(f"Average DRG codes per record: {evaluation_df['drg_codes'].str.split(',').str.len().mean():.1f}")
    
    # ICD version distribution
    version_counts = evaluation_df['primary_icd_version'].value_counts()
    print(f"\nICD Version Distribution:")
    for version, count in version_counts.items():
        print(f"  ICD-{version}: {count} records ({count/len(evaluation_df)*100:.1f}%)")
    
    # Show sample record
    print("\n=== Sample Record ===")
    sample_record = evaluation_df.iloc[0]
    print(f"HADM ID: {sample_record['hadm_id']}")
    print(f"Subject ID: {sample_record['subject_id']}")
    print(f"ICD Codes: {sample_record['icd_codes']}")
    print(f"Primary ICD Version: {sample_record['primary_icd_version']}")
    print(f"DRG Codes: {sample_record['drg_codes']}")
    print(f"Note preview: {sample_record['note_text'][:200]}...")
    
    print(f"\n✅ Successfully created evaluation datasets:")
    print(f"  CSV: {OUTPUT_CSV_PATH}")
    print(f"  JSON: {OUTPUT_JSON_PATH}")
    print("Ready for LLM medical coding evaluation with version-appropriate prompts!")

if __name__ == "__main__":
    main()
