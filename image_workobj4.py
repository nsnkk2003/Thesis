import cv2
import pandas as pd
import os

def extract_jump_the_difference():
    # --- 1. YOUR EXACT INPUTS ---
    xlsx_path = "Annom_WorkObject4.xlsx"
    output_dir = "workobject4_frames"  
    
    videos_info = [
        {
            "path": "video_data/20240521-160638190582.webm",
            "start": pd.to_datetime("2024-05-21 16:06:38.190")
        },
        {
            "path": "video_data/20240521-163107415027.webm",
            "start": pd.to_datetime("2024-05-21 16:31:07.415")
        },
        {
            "path": "video_data/20240522-101726215737.webm",
            "start": pd.to_datetime("2024-05-22 10:17:26.215")
        },
        {
            "path": "video_data/20240522-131943132767.webm",
            "start": pd.to_datetime("2024-05-22 13:19:43.132")
        },
        {
            "path": "video_data/reassembled_22.webm",
            "start": pd.to_datetime("2024-05-22 14:00:51.439")
        }
    ]

    print(f"Loading logs from {xlsx_path}...")
    df = pd.read_excel(xlsx_path)
    df['Time'] = df['Time'].astype(str).str.replace(',', '.')
    df['Time'] = pd.to_datetime(df['Time'])
    
    # Sort just to guarantee we are strictly moving forward
    df = df.sort_values(by='Time') 
    
    os.makedirs(output_dir, exist_ok=True)
    total_extracted = 0

    # --- 2. PROCESS EACH VIDEO ---
    for i, vid in enumerate(videos_info):
        vid_path = vid['path']
        vid_start = vid['start']
        
        # Dynamic window: safely assign logs to the correct video
        if i < len(videos_info) - 1:
            next_vid_start = videos_info[i+1]['start']
            vid_logs = df[(df['Time'] >= vid_start) & (df['Time'] < next_vid_start)]
        else:
            vid_logs = df[df['Time'] >= vid_start]

        if vid_logs.empty:
            continue

        print(f"\n--- Opening Video {i+1}: {os.path.basename(vid_path)} ---")
        print(f"Assigned {len(vid_logs)} logs. Jumping the differences forward...")
        
        cap = cv2.VideoCapture(vid_path)
        
        # We start our timer at the exact moment the video begins
        previous_log_time = vid_start 
        
        # --- 3. THE "JUMP THE DIFFERENCE" LOGIC ---
        for index, row in vid_logs.iterrows():
            current_log_time = row['Time']
            
            # 1. Calculate the exact difference between the last log and this log
            jump_difference_ms = (current_log_time - previous_log_time).total_seconds() * 1000.0
            
            # We calculate the absolute target to prevent micro-drifts in the math
            target_ms = (current_log_time - vid_start).total_seconds() * 1000.0
            
            if target_ms < 0:
                continue
            
            # 2. Fast-forward exactly that difference
            while True:
                current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                
                # Did our fast-forward cross the gap?
                if current_ms >= target_ms:
                    # Boom. Decode the real picture.
                    ret, frame = cap.retrieve()
                    if ret:
                        label = "pore" if row['pore_diameter'] > 0 else "normal"
                        safe_time = current_log_time.strftime("%Y-%m-%d %H-%M-%S,%f")[:-3]
                        filename = f"{safe_time}_{label}_{index}.jpg"
                        
                        cv2.imwrite(os.path.join(output_dir, filename), frame)
                        total_extracted += 1
                        
                    # Update our anchor to this log so we can calculate the NEXT difference
                    previous_log_time = current_log_time 
                    break 
                
                # 3. If we haven't crossed the gap yet, jump forward silently without decoding
                ret = cap.grab()
                if not ret:
                    break # Video ended

        cap.release()

    print(f"\nBoom! Finished. Saved {total_extracted} unique frames perfectly matched to reality.")

extract_jump_the_difference()
