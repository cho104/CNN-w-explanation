#pragma once
#include "Box.h"
#include "Utils.h"
#include "rlgl.h"

class TensorVisualizer {
public:
  Tensor4D tensor;
  Vec3 gridPos, gridSize, bSize, gap;
  float speedMin, speedMax, drag;
  std::vector<std::vector<std::vector<Box *>>> boxes;
  std::vector<Box *> boxes1D;
  int tC, tW, tH;
  std::vector<Image> heatmapImages;
  std::vector<Texture2D> heatmapTextures;
  bool showHeatmap;
  bool useRevealLimit = false;
  int revealedZ = 0, revealedY = 0, revealedX = 0;

  TensorVisualizer(Vec3 gridPos, Vec3 gridSize, Tensor4D tensor, Vec3 bSize,
                   float speedMin, float speedMax, float drag) {
    this->gridPos = gridPos;
    this->gridSize = gridSize;
    this->tensor = tensor;
    this->speedMin = speedMin;
    this->speedMax = speedMax;
    this->drag = drag;
    this->tC = tensor[0].size();
    this->tW = tensor[0][0].size();
    this->tH = tensor[0][0][0].size();
    this->bSize = bSize;

    gap = Vec3(gridSize.x / (float)std::max(1, tW - 1),
               gridSize.y / (float)std::max(1, tH - 1),
               gridSize.z / (float)std::max(1, tC - 1));
    boxes.resize(tC, std::vector<std::vector<Box *>>(
                         tW, std::vector<Box *>(tH, nullptr)));

    for (int i = 0; i < tC; i++) {
      for (int j = 0; j < tW; j++) {
        for (int k = 0; k < tH; k++) {
          float z = (tC == 1) ? gridPos.z
                              : map_val(i, 0, tC - 1, -gridSize.z / 2,
                                        gridSize.z / 2) +
                                    gridPos.z;
          float x = (tW == 1) ? gridPos.x
                              : map_val(j, 0, tW - 1, -gridSize.x / 2,
                                        gridSize.x / 2) +
                                    gridPos.x;
          float y = (tH == 1) ? gridPos.y
                              : map_val(k, 0, tH - 1, -gridSize.y / 2,
                                        gridSize.y / 2) +
                                    gridPos.y;

          // FIX: Invert Raylib Y-axis to perfectly match your 2D canvas drawing
          // layout
          Vec3 pos = (tH == 1) ? Vec3(x, -y, z) : Vec3(y, -x, z);

          float speedMapped = map_val(i * tW * tH + j * tH + k, 0, tC * tW * tH,
                                      speedMin, speedMax);

          float val = tensor[0][i][j][k];
          Vec3 initColor = valueToDivergentColor(0.0f, 2.0f); // Force start as blank white, ignore resting bias
          Box *newBox =
              new Box(pos, bSize, val, initColor, speedMapped, drag);
          boxes[i][j][k] = newBox;
          boxes1D.push_back(newBox);
        }
      }
    }

    showHeatmap = (tC > 1 && tW > 1 && tH > 1);
    if (showHeatmap) {
      heatmapImages.resize(tC);
      heatmapTextures.resize(tC);
      for (int i = 0; i < tC; i++) {
        heatmapImages[i] = GenImageColor(tW, tH, BLANK);
        heatmapTextures[i] = LoadTextureFromImage(heatmapImages[i]);
      }
    }
  }

  ~TensorVisualizer() {
    for (auto b : boxes1D)
      delete b;
    if (showHeatmap) {
      for (int i = 0; i < tC; i++) UnloadTexture(heatmapTextures[i]);
    }
  }

  void setValue(const Tensor4D &value) {
    for (int z = 0; z < tC; z++) {
      for (int y = 0; y < tH; y++) {
        for (int x = 0; x < tW; x++) {
          bool isRevealed = true;
          if (useRevealLimit) {
            if (z > revealedZ) isRevealed = false;
            else if (z == revealedZ && y > revealedY) isRevealed = false;
            else if (z == revealedZ && y == revealedY && x > revealedX) isRevealed = false;
          }
          
          if (!isRevealed) {
            boxes[z][x][y]->trgValue = Vec3(1.0f, 1.0f, 1.0f);
            if (showHeatmap) {
              ImageDrawPixel(&heatmapImages[z], x, y, BLANK);
            }
            continue;
          }

          float v = value[0][z][x][y];
          boxes[z][x][y]->trgValue = valueToDivergentColor(v, 2.0f);
          boxes[z][x][y]->displayValue = v;
          boxes[z][x][y]->computed = true;
          
          if (showHeatmap) {
            Vec3 cVec = valueToDivergentColor(v, 2.0f);
            Color c = {(unsigned char)std::clamp(cVec.x * 255.0f, 0.0f, 255.0f),
                       (unsigned char)std::clamp(cVec.y * 255.0f, 0.0f, 255.0f),
                       (unsigned char)std::clamp(cVec.z * 255.0f, 0.0f, 255.0f), 200};
            ImageDrawPixel(&heatmapImages[z], x, y, c);
          }
        }
      }
      if (showHeatmap) {
        UpdateTexture(heatmapTextures[z], heatmapImages[z].data);
      }
    }
  }

  void clearValue() {
    Vec3 whiteColor = valueToDivergentColor(0.0f, 2.0f);
    for (auto b : boxes1D) {
      b->trgValue = whiteColor;
      b->displayValue = 0.0f;
      b->computed = false;
    }
    if (showHeatmap) {
      for (int z = 0; z < tC; z++) {
        ImageClearBackground(&heatmapImages[z], BLANK);
        UpdateTexture(heatmapTextures[z], heatmapImages[z].data);
      }
    }
  }

  void draw() {
    for (auto b : boxes1D) {
      b->update();
      b->draw();
    }
    
    if (showHeatmap) {
      for (int z = 0; z < tC; z++) {
        if (heatmapTextures[z].id == 0) continue;
        
        float z_pos = map_val(z, 0, tC - 1, -gridSize.z / 2, gridSize.z / 2) + gridPos.z;
        float zFront = z_pos + 0.1f;
        
        float minX_grid = -gridSize.x / 2 + gridPos.x;
        float maxX_grid =  gridSize.x / 2 + gridPos.x;
        float minY_grid = -gridSize.y / 2 + gridPos.y;
        float maxY_grid =  gridSize.y / 2 + gridPos.y;
        
        Vec3 pTL(minY_grid, -minX_grid, zFront);
        Vec3 pBL(maxY_grid, -minX_grid, zFront);
        Vec3 pBR(maxY_grid, -maxX_grid, zFront);
        Vec3 pTR(minY_grid, -maxX_grid, zFront);
        
        rlSetTexture(heatmapTextures[z].id);
        rlBegin(RL_QUADS);
        rlColor4ub(255, 255, 255, 255);
        
        rlTexCoord2f(0.0f, 0.0f); rlVertex3f(pTL.x, pTL.y, pTL.z);
        rlTexCoord2f(0.0f, 1.0f); rlVertex3f(pBL.x, pBL.y, pBL.z);
        rlTexCoord2f(1.0f, 1.0f); rlVertex3f(pBR.x, pBR.y, pBR.z);
        rlTexCoord2f(1.0f, 0.0f); rlVertex3f(pTR.x, pTR.y, pTR.z);
        
        rlEnd();
        rlSetTexture(0);
      }
    }
  }

  void setTargetPos(TensorVisualizer *tv) {
    for (size_t i = 0; i < boxes1D.size(); i++) {
      boxes1D[i]->setTarget(*tv->boxes1D[i]);
      boxes1D[i]->curValue = tv->boxes1D[i]->curValue;
      boxes1D[i]->trgValue = tv->boxes1D[i]->trgValue;
    }
  }

  void clearHighlights() {
    for (auto b : boxes1D) {
      b->highlighted = false;
    }
  }

  // Compute bounding box from actual box positions (handles all coordinate transforms correctly)
  BoundingBox getOverallBoundingBox() const {
    if (boxes1D.empty()) return {{0,0,0},{0,0,0}};
    float minX = 1e9f, minY = 1e9f, minZ = 1e9f;
    float maxX = -1e9f, maxY = -1e9f, maxZ = -1e9f;
    for (auto b : boxes1D) {
      float hx = std::max(b->curBSize.x, b->trgBSize.x) * 0.5f + 0.5f;
      float hy = std::max(b->curBSize.y, b->trgBSize.y) * 0.5f + 0.5f;
      float hz = std::max(b->curBSize.z, b->trgBSize.z) * 0.5f + 0.5f;
      Vec3 p = b->curPos;
      if (p.x - hx < minX) minX = p.x - hx;
      if (p.y - hy < minY) minY = p.y - hy;
      if (p.z - hz < minZ) minZ = p.z - hz;
      if (p.x + hx > maxX) maxX = p.x + hx;
      if (p.y + hy > maxY) maxY = p.y + hy;
      if (p.z + hz > maxZ) maxZ = p.z + hz;
    }
    return {{minX, minY, minZ}, {maxX, maxY, maxZ}};
  }
};