import os
import shutil

# CONFIGURATION
SOURCE_DIR = "data/external/UTA-RLDD-RAW" 
DEST_DIR = "data/external/videos"

def organize_dataset():
    # Create the 3 class folders
    for category in ['attentive', 'low_vigilance', 'drowsy']:
        os.makedirs(os.path.join(DEST_DIR, category), exist_ok=True)
    
    print(f"Scanning {SOURCE_DIR} for videos...")
    
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Could not find {SOURCE_DIR}")
        return

    count = 0
    # Walk through all folders
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            file_lower = file.lower()
            
            # Check if it's a video file
            if file_lower.endswith(('.mp4', '.mov', '.avi')):
                
                category = None
                name, _ = os.path.splitext(file_lower) 
                
                # BULLETPROOF MATCHING
                if "alert" in name or name.endswith("0") or name == "0":
                    category = 'attentive'
                elif "vigilance" in name or "low" in name or name.endswith("5") or name == "5":
                    category = 'low_vigilance'
                elif "drowsy" in name or "drowsiness" in name or name.endswith("10") or name == "10":
                    category = 'drowsy'
                
                if category:
                    src_path = os.path.join(root, file)
                    
                    # FIX: Prevent Overwriting by adding the parent folder name!
                    parent_folder_name = os.path.basename(root) 
                    new_filename = f"{parent_folder_name}_{file}"
                    
                    dst_path = os.path.join(DEST_DIR, category, new_filename)
                    
                    shutil.copy2(src_path, dst_path)
                    print(f"Moved: {parent_folder_name}/{file} -> {category}/{new_filename}")
                    count += 1

    print(f"\nSUCCESS: Organized {count} videos.")

if __name__ == "__main__":
    organize_dataset()