#pragma once
#include "Utils.h"

class MLP {
public:
  Tensor2D weights;
  Tensor1D bias;
  bool useRelu;

  MLP(const std::vector<std::string> &weightsStr,
      const std::vector<std::string> &biasStr, bool reluFlag) {
    useRelu = reluFlag;
    weights = parse2dTensor(weightsStr);
    bias = parseConvBias(biasStr);
  }

  Tensor2D parse2dTensor(const std::vector<std::string> &tensorStr) {
    int batchNum = tensorStr.size();
    int inChNum = split(tensorStr[0], " ").size();
    Tensor2D tensorParsed(batchNum, Tensor1D(inChNum, 0.0f));
    for (int i = 0; i < batchNum; i++) {
      auto inCh = split(tensorStr[i], " ");
      for (int j = 0; j < inChNum; j++)
        tensorParsed[i][j] = std::stof(inCh[j]);
    }
    return tensorParsed;
  }

  Tensor1D parseConvBias(const std::vector<std::string> &biasStr) {
    int outChNum = biasStr.size();
    Tensor1D weightsParsed(outChNum);
    for (int i = 0; i < outChNum; i++)
      weightsParsed[i] = std::stof(biasStr[i]);
    return weightsParsed;
  }

  Tensor2D forward(const Tensor2D &x) {
    int m1Rows = weights.size();
    int m1Cols = weights[0].size();
    int m2Cols = x[0].size();

    Tensor2D result(m1Rows, Tensor1D(m2Cols, 0.0f));
    for (int i = 0; i < m1Rows; i++) {
      for (int j = 0; j < m2Cols; j++) {
        float sum = 0.0f;
        for (int k = 0; k < m1Cols; k++) {
          sum += weights[i][k] * x[k][j];
        }
        result[i][j] = sum;
      }
      result[i][0] += bias[i];
      if (useRelu)
        result[i][0] = relu(result[i][0]);
    }
    return result;
  }
};