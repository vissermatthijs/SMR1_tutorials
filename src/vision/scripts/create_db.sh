#!/bin/bash

#Here we are running a VIS top-view pipeline
#"-d", "--dir", help='Input directory containing images or snapshots.', required=True
#"-p", "--pipeline", help='Pipeline script file.', required=True
#"-s", "--db", help='SQLite database file name.', required=True
#"-t", "--type", help='Image format type (extension).', default="png"
#"-l", "--delimiter", help='Image file name metadata delimiter character.', default='_'
#"-f", "--meta",help='Image file name metadata format. List valid metadata fields separated by the ''delimiter (-l/--delimiter). Valid metadata fields are: ' + ', '.join(map(str, list(valid_meta.keys()))), default='imgtype_camera_frame_zoom_id'


time \
/home/matthijs/plantcv/plantcv-pipeline.py \
-d /home/matthijs/PycharmProjects/SMR1/src/vision/scripts/yucca_rename/yucca1 \
-p /home/matthijs/PycharmProjects/SMR1/src/vision/scripts/feature_extract.py \
-a filename \
-s plant_db_1_v5 \
-f camera_timestamp_id_other \
-T 10