"""
Creates a proper sample dataset with discharge summaries AND corresponding ICD/DRG codes.
Based on josh.py approach - ensures we only sample hadm_ids that have both notes and codes.
"""
import pandas as pd
import random
from pathlib import Path

def create_sample_dataset():
    # Paths
    base_path = Path("/home/justjosh/Turing-Test/src/data/mimic")
    
    print("Loading MIMIC data files...")
    
    # Load all required files
    notes = pd.read_csv(base_path / "NOTEEVENTS.csv", low_memory=False)
    diagnoses = pd.read_csv(base_path / "hosp" / "diagnoses_icd.csv")
    drg = pd.read_csv(base_path / "hosp" / "drgcodes.csv")
    d_icd_diag = pd.read_csv(base_path / "hosp" / "d_icd_diagnoses.csv")
    
    print(f"Notes shape: {notes.shape}")
    print(f"Diagnoses shape: {diagnoses.shape}")
    print(f"DRG shape: {drg.shape}")
    
    # Normalize column names to lowercase
    notes.columns = notes.columns.str.lower()
    diagnoses.columns = diagnoses.columns.str.lower()
    drg.columns = drg.columns.str.lower()
    d_icd_diag.columns = d_icd_diag.columns.str.lower()
    
    # Filter discharge summaries
    discharge_notes = notes[notes["category"] == "Discharge summary"].copy()
    print(f"Discharge notes found: {len(discharge_notes)}")
    
    # Remove duplicates, keep last per hadm_id
    discharge_notes = discharge_notes.drop_duplicates(subset=["hadm_id"], keep="last")
    print(f"Unique discharge notes: {len(discharge_notes)}")
    
    # Get hadm_ids that have both ICD codes AND DRG codes
    hadm_with_icd = set(diagnoses['hadm_id'].unique())
    hadm_with_drg = set(drg['hadm_id'].unique())
    hadm_with_notes = set(discharge_notes['hadm_id'].dropna())
    
    # Find intersection - hadm_ids that have ALL three: notes, ICD codes, AND DRG codes
    hadm_with_all = hadm_with_notes.intersection(hadm_with_icd).intersection(hadm_with_drg)
    print(f"HADMs with notes: {len(hadm_with_notes)}")
    print(f"HADMs with ICD codes: {len(hadm_with_icd)}")
    print(f"HADMs with DRG codes: {len(hadm_with_drg)}")
    print(f"HADMs with ALL (notes + ICD + DRG): {len(hadm_with_all)}")
    
    if len(hadm_with_all) < 20:
        print(f"WARNING: Only {len(hadm_with_all)} hadm_ids have all required data")
        sample_size = len(hadm_with_all)
    else:
        sample_size = 20
    
    # Sample from those that have all data
    sampled_hadm_ids = random.Random(42).sample(list(hadm_with_all), sample_size)
    print(f"Sampling {sample_size} hadm_ids: {sampled_hadm_ids[:5]}...")
    
    # Get the sampled discharge notes
    sampled_notes = discharge_notes[discharge_notes['hadm_id'].isin(sampled_hadm_ids)].copy()
    
    # Create ICD codes mapping (using actual codes, not descriptions)
    icd_grouped = diagnoses.groupby('hadm_id')['icd_code'].apply(list).reset_index()
    icd_grouped.columns = ['hadm_id', 'icd_codes']
    
    # Create DRG codes mapping
    drg_grouped = drg.groupby('hadm_id')['drg_code'].apply(list).reset_index()
    drg_grouped.columns = ['hadm_id', 'drg_codes']
    
    # Merge everything together
    final_dataset = (sampled_notes[['hadm_id', 'subject_id', 'text']]
                    .merge(icd_grouped, on='hadm_id', how='left')
                    .merge(drg_grouped, on='hadm_id', how='left'))
    
    # Convert lists to comma-separated strings for CSV storage
    final_dataset['icd_codes'] = final_dataset['icd_codes'].apply(lambda x: ','.join(map(str, x)) if x else '')
    final_dataset['drg_codes'] = final_dataset['drg_codes'].apply(lambda x: ','.join(map(str, x)) if x else '')
    
    # Rename text column to note_text for consistency
    final_dataset = final_dataset.rename(columns={'text': 'note_text'})
    
    # Save the result
    output_path = base_path / "sampled_discharge_summaries_with_codes.csv"
    final_dataset.to_csv(output_path, index=False)
    
    print(f"\n✅ Created sample dataset with {len(final_dataset)} samples")
    print(f"Saved to: {output_path}")
    
    # Show sample statistics
    print(f"\nSample statistics:")
    print(f"- Samples with ICD codes: {(final_dataset['icd_codes'] != '').sum()}")
    print(f"- Samples with DRG codes: {(final_dataset['drg_codes'] != '').sum()}")
    print(f"- Average ICD codes per sample: {final_dataset['icd_codes'].str.split(',').str.len().mean():.1f}")
    print(f"- Average DRG codes per sample: {final_dataset['drg_codes'].str.split(',').str.len().mean():.1f}")
    
    # Show first few samples
    print(f"\nFirst 3 samples:")
    for i, row in final_dataset.head(3).iterrows():
        print(f"HADM_ID {row['hadm_id']}: {len(row['icd_codes'].split(','))} ICD codes, {len(row['drg_codes'].split(','))} DRG codes")
        print(f"  ICD: {row['icd_codes'][:100]}...")
        print(f"  DRG: {row['drg_codes']}")
        print(f"  Note length: {len(row['note_text'])} chars")
        print()
    
    return final_dataset

if __name__ == "__main__":
    create_sample_dataset()
