#pragma once
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

typedef std::vector<float> Tensor1D;
typedef std::vector<Tensor1D> Tensor2D;
typedef std::vector<Tensor2D> Tensor3D;
typedef std::vector<Tensor3D> Tensor4D;

inline Tensor4D create4D(int d1, int d2, int d3, int d4, float val = 0.0f) {
  return Tensor4D(d1, Tensor3D(d2, Tensor2D(d3, Tensor1D(d4, val))));
}

inline std::vector<std::string> loadStrings(const std::string &path) {
  std::vector<std::string> lines;
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "Could not open file: " << path << std::endl;
    return lines;
  }
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty())
      lines.push_back(line);
  }
  return lines;
}

inline std::vector<std::string> split(const std::string &str,
                                      const std::string &delimiter) {
  std::vector<std::string> tokens;
  size_t prev = 0, pos = 0;
  while ((pos = str.find(delimiter, prev)) != std::string::npos) {
    if (pos > prev)
      tokens.push_back(str.substr(prev, pos - prev));
    prev = pos + delimiter.length();
  }
  if (prev < str.length())
    tokens.push_back(str.substr(prev));
  return tokens;
}

inline float relu(float x) { return std::max(x, 0.0f); }

inline float map_val(float value, float istart, float istop, float ostart,
                     float ostop) {
  if (istop - istart == 0)
    return ostart;
  return ostart + (ostop - ostart) * ((value - istart) / (istop - istart));
}

inline void printTensor(const Tensor2D &tensor) {
  int outChNum = tensor.size();
  int inChNum = tensor[0].size();
  for (int i = 0; i < outChNum; i++) {
    std::cout << std::endl;
    for (int j = 0; j < inChNum; j++) {
      std::cout << tensor[i][j] << " ";
    }
  }
  std::cout << "\nThe shape of this image is : " << outChNum << ", " << inChNum
            << std::endl;
}

inline Tensor2D flatten(const Tensor4D &tensor) {
  std::vector<float> arrList;
  for (size_t i = 0; i < tensor.size(); i++) {
    for (size_t j = 0; j < tensor[0].size(); j++) {
      for (size_t k = 0; k < tensor[0][0].size(); k++) {
        for (size_t l = 0; l < tensor[0][0][0].size(); l++) {
          arrList.push_back(tensor[i][j][k][l]);
        }
      }
    }
  }
  Tensor2D floatArray(arrList.size(), Tensor1D(1, 0.0f));
  for (size_t i = 0; i < arrList.size(); i++) {
    floatArray[i][0] = arrList[i];
  }
  return floatArray;
}

inline Tensor2D softmax(const Tensor2D &x) {
  Tensor2D val(x.size(), Tensor1D(1, 0.0f));
  float div = 0;
  for (size_t i = 0; i < x.size(); i++) {
    float tempX = x[i][0];
    val[i][0] = std::exp(tempX);
    div += std::exp(tempX);
  }
  for (size_t i = 0; i < x.size(); i++)
    val[i][0] /= div;
  return val;
}

inline Tensor4D parse4dTensor(const std::vector<std::string> &tensorStr) {
  int outChNum = tensorStr.size();
  int inChNum = 0, kernelWNum = 0, kernelHNum = 0;
  for (int i = 0; i < outChNum; i++) {
    auto inCh = split(tensorStr[i], "!");
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
  Tensor4D tensorParsed(
      outChNum,
      Tensor3D(inChNum, Tensor2D(kernelWNum, Tensor1D(kernelHNum, 0.0f))));
  for (int i = 0; i < outChNum; i++) {
    auto inCh = split(tensorStr[i], "!");
    for (int j = 0; j < inChNum; j++) {
      auto kernelW = split(inCh[j], ",");
      for (int k = 0; k < kernelWNum; k++) {
        auto kernelH = split(kernelW[k], " ");
        for (int l = 0; l < kernelHNum; l++) {
          tensorParsed[i][j][k][l] = std::stof(kernelH[l]);
        }
      }
    }
  }
  return tensorParsed;
}

#include "Vec3.h"

inline Vec3 valueToDivergentColor(float v, float maxVal = 1.0f) {
  float t = std::clamp(v / maxVal, -1.0f, 1.0f);
  if (t < 0.0f) {
    float intensity = 1.0f + t;
    return Vec3(intensity, intensity, 1.0f);
  } else {
    float intensity = 1.0f - t;
    return Vec3(1.0f, intensity, intensity);
  }
}