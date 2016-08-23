#ifndef DETECTION_CONVERSION_H
#define DETECTION_CONVERSION_H

#include "box.h"

void convert_detections(float *predictions, 
			int classes, 
			int num, 
			int square, 
			int side, 
			int w, 
			int h, 
			float thresh, 
			float **probs, 
			box *boxes, 
			int only_objectness);

#endif