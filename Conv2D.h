#pragma once
#include "Utils.h"

class Conv2D {
public:
  Tensor4D weights;
  int wShape[4];
  Tensor1D bias;
  int stride = 2;

  Conv2D(const std::vector<std::string> &weightsStr,
         const std::vector<std::string> &biasStr) {
    bias = parseConvBias(biasStr);
    weights = parseConvWeights(weightsStr);
  }

  Tensor1D parseConvBias(const std::vector<std::string> &biasStr) {
    int outChNum = biasStr.size();
    Tensor1D weightsParsed(outChNum);
    for (int i = 0; i < outChNum; i++)
      weightsParsed[i] = std::stof(biasStr[i]);
    return weightsParsed;
  }

  Tensor4D parseConvWeights(const std::vector<std::string> &weightsStr) {
    int outChNum = weightsStr.size();
    int inChNum = 0, kernelWNum = 0, kernelHNum = 0;
    for (int i = 0; i < outChNum; i++) {
      auto inCh = split(weightsStr[i], "!");
      inChNum = inCh.size();
      for (int j = 0; j < inChNum; j++) {
        auto kernelW = split(inCh[j], ",");
        kernelWNum = kernelW.size();
        for (int k = 0; k < kernelWNum; k++) {
          auto kernelH = split(kernelW[k], " ");
          kernelHNum = kernelH.size();
        }
      }
    }

    Tensor4D weightsParsed =
        create4D(outChNum, inChNum, kernelWNum, kernelHNum, 0.0f);
    for (int i = 0; i < outChNum; i++) {
      auto inCh = split(weightsStr[i], "!");
      for (int j = 0; j < inChNum; j++) {
        auto kernelW = split(inCh[j], ",");
        for (int k = 0; k < kernelWNum; k++) {
          auto kernelH = split(kernelW[k], " ");
          for (int l = 0; l < kernelHNum; l++) {
            weightsParsed[i][j][k][l] = std::stof(kernelH[l]);
          }
        }
      }
    }
    wShape[0] = outChNum;
    wShape[1] = inChNum;
    wShape[2] = kernelWNum;
    wShape[3] = kernelHNum;
    return weightsParsed;
  }

  Tensor4D forward(const Tensor4D &x) {
    int bSize = x.size();
    int imgSize = x[0][0].size() / 2;
    int inShape[] = {bSize, (int)x[0].size(), (int)x[0][0].size(),
                     (int)x[0][0][0].size()};
    int kernelH = wShape[2], kernelW = wShape[3];
    int kernelHSize = kernelH / 2, kernelWSize = kernelW / 2;

    Tensor4D result = create4D(bSize, wShape[0], imgSize, imgSize, 0.0f);

    for (int i = 0; i < wShape[0]; i++) {
      for (int j = 0; j < imgSize; j++) {
        for (int k = 0; k < imgSize; k++) {
          for (int l = 0; l < wShape[1]; l++) {
            for (int m = 0; m < kernelH; m++) {
              for (int n = 0; n < kernelW; n++) {
                int offsetX = m - kernelHSize;
                int offsetY = n - kernelWSize;
                int newPosX = j * 2 + offsetX;
                int newPosY = k * 2 + offsetY;
                if (!(newPosX < 0 || newPosY < 0 || newPosX >= inShape[2] ||
                      newPosY >= inShape[3])) {
                  result[0][i][j][k] +=
                      x[0][l][j * stride + offsetX][k * stride + offsetY] *
                      weights[i][l][m][n];
                }
              }
            }
          }
          result[0][i][j][k] += bias[i];
          result[0][i][j][k] = relu(result[0][i][j][k]);
        }
      }
    }
    return result;
  }
};