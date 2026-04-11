import pandas as pd
import glob
import os

CHUNKS_DIR = r'data\processed\v3_chunks\*.csv'
OUTPUT_FILE = r'data\processed\master_v3_dataset_with_subjects.csv'

all_files = glob.glob(CHUNKS_DIR)
print(f"Found {len(all_files)} chunks. Merging and extracting human identities...")

df_list = []
for file in all_files:
    temp_df = pd.read_csv(file)
    
    # Extract the actual Human ID from the filename
    filename = os.path.basename(file)
    if filename.startswith('UTA'):
        # Turns "UTA_42_Attentive.csv" into "UTA_42"
        parts = filename.split('_')
        human_id = f"{parts[0]}_{parts[1]}"
    elif filename.startswith('YAWDD'):
        # Turns "YAWDD_1-MaleNoGlasses-Talking.csv" into "YAWDD_1-MaleNoGlasses"
        parts = filename.split('-')
        human_id = f"{parts[0]}-{parts[1]}"
    else:
        human_id = filename
        
    temp_df['human_name'] = human_id
    df_list.append(temp_df)

master_df = pd.concat(df_list, ignore_index=True)
master_df = master_df[master_df['target'] != 'target'].dropna()

# Convert the text names into strict numeric IDs for the AI's GroupSplitter
master_df['subject_id'] = pd.factorize(master_df['human_name'])[0]
master_df = master_df.drop('human_name', axis=1) # Hide the text name from the AI

master_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n[SUCCESS] Master Dataset created with strict HUMAN Subject IDs!")
print(f"Saved to: {OUTPUT_FILE}")