#pragma once
#include "TensorVisualizer.h"

extern int globalFrameCount;

class ConvVisualizer {
public:
  TensorVisualizer *filter;
  TensorVisualizer *tv1;
  TensorVisualizer *tv2;
  Tensor4D weights;
  Tensor1D bias;
  Tensor4D *inputTensorRef;
  Tensor4D *outputTensorRef;
  
  int tC, kernelSize = 3, padding = 1, stride = 2;
  int trgX = 0, trgY = 0, trgZ = 0;
  int interval, intervalShift, lastFrame = -1;

  ConvVisualizer(TensorVisualizer *tv1, TensorVisualizer *tv2, Vec3 bSize,
                 Vec3 offset, int interval, int intervalShift, Conv2D *convLayer, 
                 Tensor4D *inputTensorRef, Tensor4D *outputTensorRef) {
    this->tv1 = tv1;
    this->tv2 = tv2;
    this->weights = convLayer->weights;
    this->bias = convLayer->bias;
    this->inputTensorRef = inputTensorRef;
    this->outputTensorRef = outputTensorRef;
    this->tC = tv1->tC;
    this->interval = interval;
    this->intervalShift = intervalShift;
    trgX += (int)(offset.x / stride);
    trgY += (int)(offset.y / stride);
    trgZ += (int)offset.z;
    if (tv2->tW > 0) trgX %= tv2->tW;
    if (tv2->tH > 0) trgY %= tv2->tH;
    if (tv2->tC > 0) trgZ %= tv2->tC;

    filter = new TensorVisualizer(Vec3(0, 0, 0), Vec3(1, 1, 1),
                                  create4D(1, tC, kernelSize, kernelSize, 0.0f),
                                  bSize, 0.3f, 0.3f, 0.4f);
  }
  ~ConvVisualizer() { delete filter; }

  void setFilterTrgPosition(int x, int y) {
    int ksh = kernelSize / 2;
    for (int i = 0; i < tC; i++) {
      Box *cb = tv1->boxes[i][x][y];
      for (int j = -ksh; j < ksh + 1; j++) {
        for (int k = -ksh; k < ksh + 1; k++) {
          int newX = j + ksh, newY = k + ksh;
          filter->boxes[i][newX][newY]->trgPos.x = cb->curPos.x + (float)j * tv1->gap.x;
          filter->boxes[i][newX][newY]->trgPos.y = cb->curPos.y + (float)k * tv1->gap.y;
          filter->boxes[i][newX][newY]->trgPos.z = cb->curPos.z;
          
          int newXtv = j + x, newYtv = k + y;
          if (newXtv >= tv1->tW || newXtv < 0 || newYtv >= tv1->tH || newYtv < 0) {
            filter->boxes[i][newX][newY]->curValue = Vec3(1.0f, 1.0f, 1.0f);
            filter->boxes[i][newX][newY]->trgValue = Vec3(1.0f, 1.0f, 1.0f);
          } else {
            tv1->boxes[i][newXtv][newYtv]->curValue = Vec3(10.0f, 10.0f, 10.0f);
            
            float avgWeight = 0.0f;
            for (int z = 0; z < tv2->tC; z++) {
                avgWeight += weights[z][i][j + ksh][k + ksh];
            }
            avgWeight /= (float)tv2->tC;
            
            filter->boxes[i][newX][newY]->trgValue = valueToDivergentColor(avgWeight, 0.5f);
          }
        }
      }
    }
  }

  void draw() {
    if (globalFrameCount != lastFrame) {
      int frameCountNew = (globalFrameCount + intervalShift) % interval;
      if (frameCountNew == 0) {
        if (trgY < tv2->tH) {
          int srcX = trgX * stride;
          int srcY = trgY * stride;
          setFilterTrgPosition(srcX, srcY);
          
          int ksh = kernelSize / 2;
          for (int z = 0; z < tv2->tC; z++) {
            float sum = bias[z];
            for (int i = 0; i < tC; i++) {
              for (int j = -ksh; j <= ksh; j++) {
                for (int k = -ksh; k <= ksh; k++) {
                  int inX = srcX + j;
                  int inY = srcY + k;
                  if (inX >= 0 && inY >= 0 && inX < tv1->tW && inY < tv1->tH) {
                     sum += (*inputTensorRef)[0][i][inX][inY] * weights[z][i][j + ksh][k + ksh];
                  }
                }
              }
            }
            sum = relu(sum);
            (*outputTensorRef)[0][z][trgX][trgY] = sum;
            
            tv2->boxes[z][trgX][trgY]->trgValue = valueToDivergentColor(sum, 2.0f);
            tv2->boxes[z][trgX][trgY]->displayValue = sum;
            tv2->boxes[z][trgX][trgY]->computed = true;
            tv2->boxes[z][trgX][trgY]->curBSize = Vec3(0, 0, 0); // pop animation
            
            if (tv2->showHeatmap) {
               Vec3 cVec = valueToDivergentColor(sum, 2.0f);
               Color c = {(unsigned char)std::clamp(cVec.x * 255.0f, 0.0f, 255.0f),
                          (unsigned char)std::clamp(cVec.y * 255.0f, 0.0f, 255.0f),
                          (unsigned char)std::clamp(cVec.z * 255.0f, 0.0f, 255.0f), 200};
               ImageDrawPixel(&tv2->heatmapImages[z], trgX, trgY, c);
               UpdateTexture(tv2->heatmapTextures[z], tv2->heatmapImages[z].data);
            }
          }
          
          trgX++;
          if (trgX >= tv2->tW) {
            trgX = 0;
            trgY++;
            if (trgY >= tv2->tH) {
              trgZ = tv2->tC; // Signal done
            }
          }
        }
      }
      lastFrame = globalFrameCount;
    }
  }
  
  void reset() {
     trgX = 0;
     trgY = 0;
     trgZ = 0;
  }
};