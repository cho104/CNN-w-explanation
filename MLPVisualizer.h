#pragma once
#include "TensorVisualizer.h"
#include "MLP.h"

extern int globalFrameCount;

class MLPVisualizer {
public:
  TensorVisualizer *tv1;
  TensorVisualizer *tv2;
  MLP *mlp;
  Tensor2D *inputTensorRef;
  Tensor2D *outputTensorRef;
  int trgNode = 0;
  int interval, intervalShift, lastFrame = -1;
  bool isSoftmax;

  MLPVisualizer(TensorVisualizer *tv1, TensorVisualizer *tv2, int interval, int intervalShift, MLP *mlp, Tensor2D *inputTensorRef, Tensor2D *outputTensorRef, bool isSoftmax) {
    this->tv1 = tv1;
    this->tv2 = tv2;
    this->mlp = mlp;
    this->interval = interval;
    this->intervalShift = intervalShift;
    this->inputTensorRef = inputTensorRef;
    this->outputTensorRef = outputTensorRef;
    this->isSoftmax = isSoftmax;
  }

  void draw() {
    if (globalFrameCount != lastFrame) {
      int frameCountNew = (globalFrameCount + intervalShift) % interval;
      if (frameCountNew == 0) {
        if (trgNode < tv2->tW) {
          // Calculate dot product for trgNode
          float sum = mlp->bias[trgNode];
          for (int i = 0; i < tv1->tW; i++) {
             sum += (*inputTensorRef)[i][0] * mlp->weights[trgNode][i];
          }
          
          if (!isSoftmax) {
             sum = relu(sum);
          }
          (*outputTensorRef)[trgNode][0] = sum;
          
          // Update tv2 dynamically!
          tv2->boxes[0][trgNode][0]->trgValue = valueToDivergentColor(sum, 2.0f);
          tv2->boxes[0][trgNode][0]->displayValue = sum;
          tv2->boxes[0][trgNode][0]->computed = true;
          tv2->boxes[0][trgNode][0]->curBSize = Vec3(0, 0, 0); // pop animation
          
          trgNode++;
        }
        
        // If we reached the end, apply softmax (if needed) and signal complete
        if (trgNode == tv2->tW) {
           if (isSoftmax) {
              Tensor2D result = softmax(*outputTensorRef);
              for (int i = 0; i < tv2->tW; i++) {
                 (*outputTensorRef)[i][0] = result[i][0];
                  tv2->boxes[0][i][0]->trgValue = valueToDivergentColor(result[i][0], 2.0f);
                  tv2->boxes[0][i][0]->displayValue = result[i][0];
                  tv2->boxes[0][i][0]->computed = true;
              }
           }
           trgNode++; // Increment to signal complete so it doesn't run every frame
        }
      }
      lastFrame = globalFrameCount;
    }
  }
  
  void reset() {
    trgNode = 0;
  }
};
