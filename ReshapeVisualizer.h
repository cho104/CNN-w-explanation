#pragma once
#include "TensorVisualizer.h"

extern int globalFrameCount;

class ReshapeVisualizer {
public:
  TensorVisualizer *tvSrc;
  TensorVisualizer *tvTrg;
  TensorVisualizer *tv;
  bool isFlattening = false;
  int lastFrame = -1;

  ReshapeVisualizer(TensorVisualizer *tvSrc, TensorVisualizer *tvTrg) {
    this->tvSrc = tvSrc;
    this->tvTrg = tvTrg;
    this->tv = new TensorVisualizer(
        tvSrc->gridPos, tvSrc->gridSize, tvSrc->tensor, tvSrc->bSize,
        tvSrc->speedMin, tvSrc->speedMax, tvSrc->drag);
  }
  ~ReshapeVisualizer() { delete tv; }

  void draw() {
    tv->draw();

    if (globalFrameCount != lastFrame) {
      if (!isFlattening) {
        tv->setTargetPos(tvSrc);
      } else {
        tv->setTargetPos(tvTrg);
      }

      lastFrame = globalFrameCount;
    }
  }
  
  void reset() {
     isFlattening = false;
  }
};
