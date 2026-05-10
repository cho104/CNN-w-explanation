#pragma once
#include "Vec3.h"
#include <algorithm>

class Box {
public:
  Vec3 curPos, trgPos, vel;
  Vec3 curBSize, trgBSize, velBSize;
  float orgValue;
  Vec3 curValue, trgValue;
  float speed, drag;

  // Interaction state
  bool highlighted = false;
  Color highlightColor = SKYBLUE;
  float displayValue = 0.0f;  // The actual computed float value for tooltip
  bool computed = false;       // True after animation has set this box's value

  Box() = default;

  Box(Vec3 pos, Vec3 bSize, float value, Vec3 initColor, float speed, float drag) {
    this->curPos = pos;
    this->trgPos = Vec3(pos.x, pos.y, pos.z);
    this->vel = Vec3();
    this->curBSize = Vec3();
    this->trgBSize = Vec3(bSize.x, bSize.y, bSize.z);
    this->velBSize = Vec3();
    this->orgValue = value;
    this->curValue = initColor;
    this->trgValue = initColor;
    this->speed = speed;
    this->drag = drag;
  }

  void draw() {
    unsigned char r =
        (unsigned char)std::clamp(curValue.x * 256.0f, 0.0f, 255.0f);
    unsigned char g =
        (unsigned char)std::clamp(curValue.y * 256.0f, 0.0f, 255.0f);
    unsigned char b =
        (unsigned char)std::clamp(curValue.z * 256.0f, 0.0f, 255.0f);
    Color col = {r, g, b, 255};

    if (highlighted) {
      // Highlighted box: brighter fill tint + colored wireframe
      Color tinted = {
        (unsigned char)std::min(255, r + 40),
        (unsigned char)std::min(255, g + 40),
        (unsigned char)std::min(255, b + 40), 255};
      DrawCube(curPos, curBSize.x, curBSize.y, curBSize.z, tinted);
      DrawCubeWires(curPos, curBSize.x + 0.1f, curBSize.y + 0.1f, curBSize.z + 0.1f, highlightColor);
    } else {
      DrawCube(curPos, curBSize.x, curBSize.y, curBSize.z, col);
      DrawCubeWires(curPos, curBSize.x + 0.05f, curBSize.y + 0.05f, curBSize.z + 0.05f, GRAY);
    }
  }

  void update() {
    curPos += (trgPos - curPos) * speed;
    velBSize += (trgBSize - curBSize) * 0.9f;
    velBSize *= 0.5f;
    curBSize += velBSize;
    curValue += (trgValue - curValue) * 0.3f;
  }

  void setTarget(const Box &b) {
    trgPos.x = b.curPos.x;
    trgPos.y = b.curPos.y;
    trgPos.z = b.curPos.z;
    trgBSize.x = b.trgBSize.x;
    trgBSize.y = b.trgBSize.y;
    trgBSize.z = b.trgBSize.z;
    curValue = b.curValue;
  }

  // Returns an axis-aligned bounding box for ray intersection testing
  BoundingBox getBoundingBox() const {
    float hx = curBSize.x * 0.5f;
    float hy = curBSize.y * 0.5f;
    float hz = curBSize.z * 0.5f;
    return {
      {curPos.x - hx, curPos.y - hy, curPos.z - hz},
      {curPos.x + hx, curPos.y + hy, curPos.z + hz}
    };
  }
};