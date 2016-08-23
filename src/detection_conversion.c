#include "detection_conversion.h"
// for pow etc.
#include <math.h>

void convert_detections(     float *predictions,
                             int classes,
                             int num /*number of predicted boxes per cell*/,
                             int square,
                             int side /*number of cells per dimension*/,
                             int w,
                             int h,
                             float thresh,
                             float **probs,
                             box *boxes,
                             int only_objectness)
{
    int i,j,n;
    //int per_cell = 5*num+classes;
    for (i = 0; i < side*side; ++i)
    {
        // cell index in x and y dimension
        int row = i / side;
        int col = i % side;
        // run over all predicted boxes for that cell
        for(n = 0; n < num; ++n)
        {
            int index     = i*num + n;
            int p_index   = side*side*classes + i*num + n;
            float scale   = predictions[p_index];
            int box_index = side*side*(classes + num) + (i*num + n)*4;

            boxes[index].x = (predictions[box_index + 0] + col) / side * w;
            boxes[index].y = (predictions[box_index + 1] + row) / side * h;
            boxes[index].w = pow(predictions[box_index + 2], (square?2:1)) * w;
            boxes[index].h = pow(predictions[box_index + 3], (square?2:1)) * h;

            for(j = 0; j < classes; ++j){
                int class_index = i*classes;
                float prob      = scale*predictions[class_index+j];
                probs[index][j] = (prob > thresh) ? prob : 0;
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
}