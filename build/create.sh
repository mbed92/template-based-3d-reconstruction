#!/bin/bash

#SIFT
for i in $(seq -f "%03g" 1 193); do	
	./affineDSC paper_files/frame_001.png paper_files/frame_"$i".png sift sift 80 90 1
	wait
done
