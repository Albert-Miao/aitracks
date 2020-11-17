Preparing the data:
    This recipe has been laid in my family for generations.
    Follow closely!
    
    1) Prep all the the other stuff:
        a) Download the video to analyze
        b) Place the raw xml of bb annotations in bb_raw
        
    2) Run parse_annotations.py on the raw annotations
        python parse_annotations -i data/bb_raw/6_annotations.xml -o data/bb/6_bb.csv
        
    3) Run frames_from_video.py on the video
        python frames_from_video.py -i data/videos/6.mp4
        
    CONDITIONAL: If there are more gps points to add to gps_1.csv,
    add them with raw_gps_to_csv.py
        python raw_gps_to_csv.py -i "data/given_coords/AI Tracks at Sea High Frequency GPS_6.txt" -o data/gps_1.csv
        
    4) Run fake_frames.py on the new camera gps log
        python fake_frames.py -i data/camera_gps_logs/SOURCE_GPS_6.csv -o data/generated_coords/6.csv
    
    5) Combine all newly generated data with collate_coords.py
        python collate_coords.py -i data/bb/6_bb.csv -o data/NN_data_skeletons/6_train.csv -c data/generated_coords/6.csv
        
Running the Data:
    Fuck if I know