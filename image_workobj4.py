import cv2
import pandas as pd
import os

def extract_exact_reality_frames():
    # --- 1. YOUR EXACT INPUTS ---
    xlsx_path = "Annom_WorkObject4.xlsx"
    output_dir = "workobject4_frames"  
    
    # Your specific video files and their true start times
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
            "path": "20240522-131943132767.webm",
            "start": pd.to_datetime("2024-05-22 13:19:43.132")
        },
        {
            "path": "video_data/reassembled_22.webm",
            "start": pd.to_datetime("2024-05-29 14:00:51.439")
        }
    ]

    print(f"Loading logs from {xlsx_path}...")
    df = pd.read_excel(xlsx_path)
    
    # Clean the Excel timestamps to make them readable by Python
    df['Time'] = df['Time'].astype(str).str.replace(',', '.')
    df['Time'] = pd.to_datetime(df['Time'])
    
    os.makedirs(output_dir, exist_ok=True)
    total_extracted = 0

    # --- 2. PROCESS EACH VIDEO ---
    for i, vid in enumerate(videos_info):
        vid_path = vid['path']
        vid_start = vid['start']
        
        # Split the Excel sheet so Video 1 gets morning logs, Video 2 gets afternoon logs
        if i == 0:
            vid_logs = df[(df['Time'] >= vid_start) & (df['Time'] < videos_info[1]['start'])]
        else:
            vid_logs = df[df['Time'] >= vid_start]

        if vid_logs.empty:
            print(f"No logs found for {os.path.basename(vid_path)}.")
            continue

        print(f"\n--- Opening Video {i+1}: {os.path.basename(vid_path)} ---")
        print(f"Found {len(vid_logs)} logs. Extracting unique frames from absolute zero...")
        
        cap = cv2.VideoCapture(vid_path)
        
        # --- 3. THE JUMP AND GRAB ---
        for index, row in vid_logs.iterrows():
            log_time = row['Time']
            
            # The Math: Find the exact millisecond offset from the start of the video
            delta_ms = (log_time - vid_start).total_seconds() * 1000.0
            
            # If the log is somehow before the video started, skip it
            if delta_ms < 0:
                continue
                
            # The Jump: Instantly move the video hardware to that exact millisecond mark
            cap.set(cv2.CAP_PROP_POS_MSEC, delta_ms)
            
            # The Grab: Pull the frame
            ret, frame = cap.read()
            
            if ret:
                # The Pore Logic: Check the diameter column
                label = "pore" if row['pore_diameter'] > 0 else "normal"
    
                # The Naming: European format YYYY-MM-DD HH-MM-SS,mmm
                safe_time = log_time.strftime("%Y-%m-%d %H-%M-%S,%f")[:-3]
                filename = f"{safe_time}_{label}.jpg"
                
                # Save the image
                cv2.imwrite(os.path.join(output_dir, filename), frame)
                total_extracted += 1

        cap.release()

    print(f"\nBoom! Finished. Saved {total_extracted} unique frames to the '{output_dir}' folder.")

# Run the script
extract_exact_reality_frames()