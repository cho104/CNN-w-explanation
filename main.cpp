#include "Conv2D.h"
#include "ConvVisualizer.h"
#include "InteractionManager.h"
#include "MLP.h"
#include "MLPVisualizer.h"
#include "ReshapeVisualizer.h"
#include "TensorVisualizer.h"
#include "raylib.h"
#include <cmath>

int globalFrameCount = 0;
bool hasInput = false;
bool flattenDone = false;
bool animationStarted = false;

Conv2D *c1, *c2, *c3, *c4;
MLP *mlp1, *mlp2;
std::vector<TensorVisualizer *> tvs;
std::vector<ConvVisualizer *> cvs;
std::vector<MLPVisualizer *> mvs;
ReshapeVisualizer *rv;
InteractionManager interactionMgr;

RenderTexture2D canvasRT;
Tensor4D inputMat;
std::vector<Tensor4D> currentOuts;
Camera3D cam = {0};

float camAngleX = -0.8f;
float camAngleY = 0.5f;
float camDist =
    200.0f; // FIX: Scaled down model distance to prevent disappearance
Vector3 camTarget = {0.0f, 0.0f, 0.0f};

void forwardNetwork() {
  Tensor4D out1 = c1->forward(inputMat);
  Tensor4D out2 = c2->forward(out1);
  Tensor4D out3 = c3->forward(out2);
  Tensor4D out4 = c4->forward(out3);
  Tensor2D out5 = flatten(out4);
  Tensor2D out6 = mlp1->forward(out5);
  Tensor2D result = mlp2->forward(out6);
  result = softmax(result);

  Tensor4D out5Temp = create4D(1, 1, 1, out5.size(), 0.0f);
  out5Temp[0][0] = out5;
  Tensor4D out6Temp = create4D(1, 1, 1, out6.size(), 0.0f);
  out6Temp[0][0] = out6;
  Tensor4D resultTemp = create4D(1, 1, 1, result.size(), 0.0f);
  resultTemp[0][0] = result;

  currentOuts = {inputMat, out1,     out2,     out3,
                 out4,     out5Temp, out6Temp, resultTemp};
}

void resetCamera() {
  camAngleX = -0.8f;
  camAngleY = 0.5f;
  camDist = 200.0f;
  camTarget = {0.0f, 0.0f, 0.0f};
}

int main() {
  SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
  InitWindow(1280, 720, "ConvNet Visualizer - C++ Raylib");
  SetTargetFPS(60);

  inputMat = create4D(1, 1, 32, 32, 0.0f);

  canvasRT = LoadRenderTexture(32, 32);
  BeginTextureMode(canvasRT);
  ClearBackground(BLACK);
  EndTextureMode();

  cam.up = {0.0f, 1.0f, 0.0f};
  cam.fovy = 45.0f;
  cam.projection = CAMERA_PERSPECTIVE;

  c1 = new Conv2D(loadStrings("conv1Weight.txt"), loadStrings("conv1Bias.txt"));
  c2 = new Conv2D(loadStrings("conv2Weight.txt"), loadStrings("conv2Bias.txt"));
  c3 = new Conv2D(loadStrings("conv3Weight.txt"), loadStrings("conv3Bias.txt"));
  c4 = new Conv2D(loadStrings("conv4Weight.txt"), loadStrings("conv4Bias.txt"));
  mlp1 =
      new MLP(loadStrings("mlp1Weight.txt"), loadStrings("mlp1Bias.txt"), true);
  mlp2 = new MLP(loadStrings("mlp2Weight.txt"), loadStrings("mlp2Bias.txt"),
                 false);

  // Initial silent run to construct memory allocations
  forwardNetwork();

  // FIX: Scaled ALL coordinates down by 10x to fix Z-buffer / Camera
  // disappearance
  Vec3 convBSize(2.0f, 2.0f, 0.5f), filterBSize(1.5f, 1.5f, 0.5f),
      mlpBsize(0.2f, 2.0f, 2.0f);
  Vec3 gridPositions[] = {Vec3(0, 0, -140), Vec3(0, 0, -100), Vec3(0, 0, -50),
                          Vec3(0, 0, 0),    Vec3(0, 0, 50),   Vec3(0, 0, 80),
                          Vec3(0, 0, 90),   Vec3(0, 0, 100)};
  Vec3 gridSizes[] = {Vec3(100, 100, 10), Vec3(40, 40, 40), Vec3(20, 20, 40),
                      Vec3(10, 10, 40),   Vec3(4, 4, 40),   Vec3(100, 1, 1),
                      Vec3(100, 1, 1),    Vec3(50, 1, 1)};
  Vec3 boxSizes[] = {convBSize, convBSize, convBSize, convBSize,
                     convBSize, mlpBsize,  mlpBsize,  convBSize};

  tvs.resize(8);
  for (int i = 0; i < 8; i++)
    tvs[i] =
        new TensorVisualizer(gridPositions[i], gridSizes[i], currentOuts[i],
                             boxSizes[i], 0.1f, 0.5f, 0.3f);

  cvs.resize(4);
  cvs[0] = new ConvVisualizer(tvs[0], tvs[1], filterBSize, Vec3(), 1, 0, c1, &currentOuts[0], &currentOuts[1]);
  cvs[1] = new ConvVisualizer(tvs[1], tvs[2], filterBSize, Vec3(), 1, 0, c2, &currentOuts[1], &currentOuts[2]);
  cvs[2] = new ConvVisualizer(tvs[2], tvs[3], filterBSize, Vec3(), 1, 0, c3, &currentOuts[2], &currentOuts[3]);
  cvs[3] = new ConvVisualizer(tvs[3], tvs[4], filterBSize, Vec3(), 1, 0, c4, &currentOuts[3], &currentOuts[4]);

  mvs.resize(2);
  mvs[0] = new MLPVisualizer(tvs[5], tvs[6], 1, 0, mlp1, (Tensor2D*)&currentOuts[5][0][0], (Tensor2D*)&currentOuts[6][0][0], false);
  mvs[1] = new MLPVisualizer(tvs[6], tvs[7], 1, 0, mlp2, (Tensor2D*)&currentOuts[6][0][0], (Tensor2D*)&currentOuts[7][0][0], true);

  rv = new ReshapeVisualizer(tvs[4], tvs[5]);

  // Initialize the interaction system
  std::vector<MLP*> mlpPtrs = {mlp1, mlp2};
  interactionMgr.init(tvs, cvs, mlpPtrs);

  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_R))
      resetCamera();

    if (IsKeyDown(KEY_LEFT))
      camAngleX -= 0.03f;
    if (IsKeyDown(KEY_RIGHT))
      camAngleX += 0.03f;
    if (IsKeyDown(KEY_UP))
      camAngleY -= 0.03f;
    if (IsKeyDown(KEY_DOWN))
      camAngleY += 0.03f;
    if (IsKeyDown(KEY_W))
      camDist -= 5.0f;
    if (IsKeyDown(KEY_S))
      camDist += 5.0f;

    if (camAngleY < -1.5f)
      camAngleY = -1.5f;
    if (camAngleY > 1.5f)
      camAngleY = 1.5f;
    if (camDist < 30.0f)
      camDist = 30.0f;

    cam.target = camTarget;
    cam.position.x = cam.target.x + camDist * cosf(camAngleY) * sinf(camAngleX);
    cam.position.y = cam.target.y + camDist * sinf(camAngleY);
    cam.position.z = cam.target.z + camDist * cosf(camAngleY) * cosf(camAngleX);

    int screenW = GetScreenWidth();
    int screenH = GetScreenHeight();

    // FIX: Clean Floating Canvas at bottom right corner
    float canvasUIW = 256.0f;
    float canvasUIX = screenW - canvasUIW - 20.0f;
    float canvasUIY = screenH - canvasUIW - 20.0f;

    Vector2 mousePos = GetMousePosition();
    Vector2 mouseDelta = GetMouseDelta();
    bool mouseOnCanvas =
        (mousePos.x > canvasUIX && mousePos.x < canvasUIX + canvasUIW &&
         mousePos.y > canvasUIY && mousePos.y < canvasUIY + canvasUIW);

    float cx1 = 32.0f * map_val(mousePos.x, canvasUIX, canvasUIX + canvasUIW,
                                0.0f, 1.0f);
    float cy1 = 32.0f * map_val(mousePos.y, canvasUIY, canvasUIY + canvasUIW,
                                0.0f, 1.0f);
    float cx2 = 32.0f * map_val(mousePos.x - mouseDelta.x, canvasUIX,
                                canvasUIX + canvasUIW, 0.0f, 1.0f);
    float cy2 = 32.0f * map_val(mousePos.y - mouseDelta.y, canvasUIY,
                                canvasUIY + canvasUIW, 0.0f, 1.0f);

    BeginTextureMode(canvasRT);
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && mouseOnCanvas) {
      hasInput = true;
      DrawLineEx({cx1, cy1}, {cx2, cy2}, 2.4f, WHITE);
    }
    if (IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_C)) {
      ClearBackground(BLACK);
      hasInput = false;
      animationStarted = false;
      globalFrameCount = 0;
      flattenDone = false;
      for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
          inputMat[0][0][i][j] = 0.0f;
          currentOuts[0][0][0][i][j] = 0.0f;
        }
      }
      for (auto cv : cvs) cv->reset();
      for (auto mv : mvs) mv->reset();
      rv->reset();
      for (int i=0; i<8; i++) tvs[i]->clearValue();
    }
    EndTextureMode();
    
    if (IsKeyPressed(KEY_ENTER) && hasInput) {
       animationStarted = true;
    }

    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && mouseOnCanvas) {
      Image img = LoadImageFromTexture(canvasRT.texture);
      Color *pixels = LoadImageColors(img);
      for (int row = 0; row < 32; row++) {
        for (int col = 0; col < 32; col++) {
          int memY = 31 - row;
          inputMat[0][0][row][col] = pixels[memY * 32 + col].r / 255.0f;
          currentOuts[0][0][0][row][col] = inputMat[0][0][row][col];
        }
      }
      UnloadImageColors(pixels);
      UnloadImage(img);
      tvs[0]->setValue(inputMat);
    }

    if (animationStarted) {
      globalFrameCount++;
      
      if (cvs[0]->trgZ < cvs[0]->tv2->tC) {
          cvs[0]->draw();
      } else if (cvs[1]->trgZ < cvs[1]->tv2->tC) {
          cvs[1]->draw();
      } else if (cvs[2]->trgZ < cvs[2]->tv2->tC) {
          cvs[2]->draw();
      } else if (cvs[3]->trgZ < cvs[3]->tv2->tC) {
          cvs[3]->draw();
      } else {
          if (!flattenDone) {
              Tensor2D out5 = flatten(currentOuts[4]);
              currentOuts[5][0][0] = out5;
              for(int i=0; i<(int)out5.size(); i++) {
                  float val = out5[i][0];
                  tvs[5]->boxes[0][i][0]->trgValue = valueToDivergentColor(val, 2.0f);
                  tvs[5]->boxes[0][i][0]->displayValue = val;
                  tvs[5]->boxes[0][i][0]->computed = true;
              }
              flattenDone = true;
              rv->isFlattening = true;
          }
          if (mvs[0]->trgNode <= mvs[0]->tv2->tW) { // use <= so softmax block can execute
              mvs[0]->draw();
          } else if (mvs[1]->trgNode <= mvs[1]->tv2->tW) {
              mvs[1]->draw();
          }
      }
    }

    // Update interaction system (hover + click detection)
    interactionMgr.update(cam, mouseOnCanvas);

    // Render entirely to window
    BeginDrawing();
    ClearBackground({15, 15, 15, 255});

    BeginMode3D(cam);
    for (auto tv : tvs)
      tv->draw();
    rv->draw();
    
    if (animationStarted) {
      for (auto cv : cvs) {
        if (cv->trgZ < cv->tv2->tC) {
           cv->filter->draw();
           break;
        }
      }
    }

    // Draw receptive-field connection lines
    interactionMgr.draw3DConnections();
    
    EndMode3D();

    // Draw hover tooltip and layer info panel (2D overlays)
    interactionMgr.drawHoverTooltip(cam);
    interactionMgr.drawLayerInfoPanel();

    // FIX: Floating Output Classified numbers positioned perfectly
    bool networkFinished = (mvs[1]->trgNode > mvs[1]->tv2->tW);
    
    for (int i = 0; i < 10; i++) {
      float xPos =
          map_val(i, 0, 9, -25.0f, 25.0f); // Matched scaled down 3D grid
      Vector2 sPos = GetWorldToScreen(Vec3(xPos, 5.0f, 100.0f), cam);

      // Draw small background for text readability
      DrawRectangle((int)sPos.x - 6, (int)sPos.y - 12, 24, 24,
                    Fade(BLACK, 0.5f));

      // Green highlight for network's predicted number
      float prob = currentOuts[7][0][0][i][0];
      Color tCol =
          (prob > 0.5f && networkFinished) ? GREEN : WHITE;
      DrawText(TextFormat("%d", i), (int)sPos.x, (int)sPos.y - 10, 20, tCol);
    }

    // Render floating UI canvas
    DrawRectangle(canvasUIX, canvasUIY, canvasUIW, canvasUIW, BLACK);
    Rectangle sourceCanv = {0.0f, 0.0f, (float)canvasRT.texture.width,
                            -(float)canvasRT.texture.height};
    DrawTexturePro(canvasRT.texture, sourceCanv,
                   {canvasUIX, canvasUIY, canvasUIW, canvasUIW}, {0, 0}, 0.0f,
                   WHITE);
    DrawRectangleLines(canvasUIX, canvasUIY, canvasUIW, canvasUIW, RAYWHITE);
    DrawText("Draw Number Here:", canvasUIX, canvasUIY - 25, 20, RAYWHITE);

    // Render HUD Details
    DrawRectangle(10, 10, 340, 150, Fade(BLACK, 0.7f));
    DrawRectangleLines(10, 10, 340, 150, DARKGRAY);
    DrawText("TRACKPAD CONTROLS:", 20, 20, 20, WHITE);
    DrawText("- Orbit Camera: ARROW KEYS", 20, 50, 16, LIGHTGRAY);
    DrawText("- Zoom Camera: [W] / [S]", 20, 70, 16, LIGHTGRAY);
    DrawText("- Draw: Trackpad/Mouse over canvas", 20, 90, 16, LIGHTGRAY);
    DrawText("- Start Animation: [ENTER]", 20, 110, 16, LIGHTGRAY);
    DrawText("- Clear Canvas: [C] or [SPACE]", 20, 130, 16, LIGHTGRAY);

    EndDrawing();
  }

  for (auto tv : tvs)
    delete tv;
  for (auto cv : cvs)
    delete cv;
  for (auto mv : mvs)
    delete mv;
  delete rv;
  delete c1;
  delete c2;
  delete c3;
  delete c4;
  delete mlp1;
  delete mlp2;

  CloseWindow();
  return 0;
}