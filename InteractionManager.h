#pragma once
#include "TensorVisualizer.h"
#include "ConvVisualizer.h"
#include "MLP.h"
#include <cmath>
#include <cstring>

// ─── Layer metadata shown in the info panel ───────────────────────────────────
struct LayerInfo {
    const char* name;
    const char* operation;
    const char* shape;
    const char* description;
    Color headerColor;
};

// ─── Centralized interaction system ───────────────────────────────────────────
class InteractionManager {
public:
    std::vector<TensorVisualizer*> tvs;
    std::vector<ConvVisualizer*>   cvs;
    std::vector<MLP*>              mlps;

    int stride    = 2;
    int kernelSize = 3;

    // Hover state
    int   hoveredLayer = -1;
    int   hoveredZ = -1, hoveredX = -1, hoveredY = -1;
    float hoveredValue = 0.0f;
    bool  hoveredComputed = false;
    Vector2 hoverScreenPos = {0, 0};

    // Click state (layer info panel)
    int selectedLayer = -1;

    // Pre-filled layer descriptions
    LayerInfo layerInfos[8];

    // Connection lines to draw
    struct ConnLine { Vec3 from; Vec3 to; bool isMLP; };
    std::vector<ConnLine> connectionLines;

    // ──────────────────────────────────────────────────────────────────────────
    InteractionManager() = default;

    void init(std::vector<TensorVisualizer*>& _tvs,
              std::vector<ConvVisualizer*>&   _cvs,
              std::vector<MLP*>&              _mlps) {
        tvs  = _tvs;
        cvs  = _cvs;
        mlps = _mlps;

        layerInfos[0] = {
            "INPUT",
            "32x32 grayscale image",
            "[1, 1, 32, 32]",
            "The raw pixel intensities drawn on the canvas,\n"
            "normalized to [0, 1]. This is the network's\n"
            "only view of the digit you drew.",
            {80, 180, 255, 255}
        };
        layerInfos[1] = {
            "CONV LAYER 1",
            "3x3 Conv, stride 2, pad 1 -> ReLU",
            "[1, 8, 16, 16]",
            "The first convolutional layer. Each of the 8\n"
            "filters slides a 3x3 window across the input\n"
            "and learns to detect simple edges and strokes.",
            {100, 220, 140, 255}
        };
        layerInfos[2] = {
            "CONV LAYER 2",
            "3x3 Conv, stride 2, pad 1 -> ReLU",
            "[1, 16, 8, 8]",
            "Combines patterns from 8 input channels into\n"
            "16 feature maps at half resolution. Detects\n"
            "mid-level features like corners and curves.",
            {255, 200, 80, 255}
        };
        layerInfos[3] = {
            "CONV LAYER 3",
            "3x3 Conv, stride 2, pad 1 -> ReLU",
            "[1, 16, 4, 4]",
            "Further reduces spatial size while deepening\n"
            "feature complexity. Detects combinations of\n"
            "curves, junctions, and partial digit shapes.",
            {255, 140, 80, 255}
        };
        layerInfos[4] = {
            "CONV LAYER 4",
            "3x3 Conv, stride 2, pad 1 -> ReLU",
            "[1, 16, 2, 2]",
            "The final convolutional layer. Compresses the\n"
            "spatial dimensions to 2x2, producing highly\n"
            "abstract representations of the input digit.",
            {255, 100, 100, 255}
        };
        layerInfos[5] = {
            "FLATTEN + FC LAYER 1",
            "Linear (64 -> 128) -> ReLU",
            "[1, 1, 1, 128]",
            "The 16 feature maps of 2x2 are flattened into\n"
            "a single 64-element vector. A fully-connected\n"
            "layer then maps it to 128 hidden units.",
            {180, 130, 255, 255}
        };
        layerInfos[6] = {
            "FC LAYER 2",
            "Linear (128 -> 10)",
            "[1, 1, 1, 10]",
            "The hidden representation is projected into 10\n"
            "raw scores (logits), one per digit class 0-9.",
            {220, 100, 220, 255}
        };
        layerInfos[7] = {
            "SOFTMAX OUTPUT",
            "Softmax normalization",
            "[1, 1, 1, 10]",
            "Converts the 10 raw logits into a probability\n"
            "distribution that sums to 1. The highest value\n"
            "is the network's predicted digit.",
            {255, 220, 80, 255}
        };
    }

    // ─── Called every frame before drawing ─────────────────────────────────────
    void update(Camera3D& cam, bool mouseOnCanvas) {
        // Clear previous frame's highlights
        for (auto tv : tvs) tv->clearHighlights();
        connectionLines.clear();
        hoveredLayer = -1;
        hoveredComputed = false;

        if (mouseOnCanvas) return;

        // Handle layer info panel click
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            int clickedLayer = -1;
            Ray ray = GetMouseRay(GetMousePosition(), cam);
            float closestClick = 1e9f;
            for (int li = 0; li < (int)tvs.size(); li++) {
                BoundingBox overallBB = tvs[li]->getOverallBoundingBox();
                if (!GetRayCollisionBox(ray, overallBB).hit) continue;
                for (auto b : tvs[li]->boxes1D) {
                    if (b->curBSize.x < 0.01f) continue;
                    BoundingBox bb = b->getBoundingBox();
                    RayCollision rc = GetRayCollisionBox(ray, bb);
                    if (rc.hit && rc.distance < closestClick) {
                        closestClick = rc.distance;
                        clickedLayer = li;
                    }
                }
            }

            if (clickedLayer >= 0) {
                selectedLayer = (selectedLayer == clickedLayer) ? -1 : clickedLayer;
            }
        }

        if (IsKeyPressed(KEY_ESCAPE)) {
            selectedLayer = -1;
        }

        // ── Hover ray-cast (two-phase) ──────────────────────────────────────
        Ray ray = GetMouseRay(GetMousePosition(), cam);
        float closestDist = 1e9f;

        for (int li = 0; li < (int)tvs.size(); li++) {
            BoundingBox overallBB = tvs[li]->getOverallBoundingBox();
            if (!GetRayCollisionBox(ray, overallBB).hit) continue;

            for (int z = 0; z < tvs[li]->tC; z++) {
                for (int x = 0; x < tvs[li]->tW; x++) {
                    for (int y = 0; y < tvs[li]->tH; y++) {
                        Box* b = tvs[li]->boxes[z][x][y];
                        if (b->curBSize.x < 0.01f) continue;
                        BoundingBox bb = b->getBoundingBox();
                        RayCollision rc = GetRayCollisionBox(ray, bb);
                        if (rc.hit && rc.distance < closestDist) {
                            closestDist = rc.distance;
                            hoveredLayer = li;
                            hoveredZ = z;
                            hoveredX = x;
                            hoveredY = y;
                        }
                    }
                }
            }
        }

        if (hoveredLayer < 0) return;

        // Mark the hovered box itself
        Box* hBox = tvs[hoveredLayer]->boxes[hoveredZ][hoveredX][hoveredY];
        hBox->highlighted = true;
        hBox->highlightColor = YELLOW;
        hoverScreenPos = GetWorldToScreen(hBox->curPos, cam);

        // Read the live display value and computed state (fix #3, #8)
        hoveredValue = hBox->displayValue;
        hoveredComputed = hBox->computed;

        // ── Receptive field tracing (conv layers 1-4) ─────────────────────
        if (hoveredLayer >= 1 && hoveredLayer <= 4) {
            int prevLayer = hoveredLayer - 1;
            int ksh = kernelSize / 2;
            int srcXCenter = hoveredX * stride;
            int srcYCenter = hoveredY * stride;
            TensorVisualizer* prevTV = tvs[prevLayer];

            for (int c = 0; c < prevTV->tC; c++) {
                for (int dx = -ksh; dx <= ksh; dx++) {
                    for (int dy = -ksh; dy <= ksh; dy++) {
                        int sx = srcXCenter + dx;
                        int sy = srcYCenter + dy;
                        if (sx < 0 || sy < 0 || sx >= prevTV->tW || sy >= prevTV->tH)
                            continue;

                        Box* srcBox = prevTV->boxes[c][sx][sy];
                        srcBox->highlighted = true;
                        srcBox->highlightColor = {0, 220, 255, 255};

                        connectionLines.push_back({srcBox->curPos, hBox->curPos, false});
                    }
                }
            }
        }

        // ── MLP layers (6-7): show top-5 strongest connections ─────────────
        if (hoveredLayer >= 6 && hoveredLayer <= 7) {
            int mlpIdx = hoveredLayer - 6;
            if (mlpIdx < (int)mlps.size()) {
                int prevLayer = hoveredLayer - 1;
                int nodeIdx = hoveredX;
                TensorVisualizer* prevTV = tvs[prevLayer];
                MLP* mlp = mlps[mlpIdx];

                if (nodeIdx < (int)mlp->weights.size()) {
                    struct WI { float absW; int idx; };
                    std::vector<WI> wis;
                    int inSize = std::min((int)mlp->weights[nodeIdx].size(), prevTV->tW);
                    for (int i = 0; i < inSize; i++) {
                        wis.push_back({std::fabs(mlp->weights[nodeIdx][i]), i});
                    }
                    std::sort(wis.begin(), wis.end(), [](const WI& a, const WI& b){ return a.absW > b.absW; });
                    int topN = std::min(5, (int)wis.size());

                    for (int t = 0; t < topN; t++) {
                        int srcIdx = wis[t].idx;
                        if (srcIdx < prevTV->tW) {
                            Box* srcBox = prevTV->boxes[0][srcIdx][0];
                            srcBox->highlighted = true;
                            srcBox->highlightColor = {0, 220, 255, 255};
                            connectionLines.push_back({srcBox->curPos, hBox->curPos, true});
                        }
                    }
                }
            }
        }
    }

    // ─── Draw connection lines in 3D space (call inside BeginMode3D) ──────────
    void draw3DConnections() {
        for (auto& cl : connectionLines) {
            if (cl.isMLP) {
                // Thicker MLP connections: draw a cylinder instead of a thin line
                Vector3 f = cl.from;
                Vector3 t = cl.to;
                DrawCylinderEx(f, t, 0.15f, 0.15f, 4, Fade(SKYBLUE, 0.5f));
            } else {
                DrawLine3D(cl.from, cl.to, Fade(SKYBLUE, 0.35f));
            }
        }
    }

    // ─── Draw hover tooltip in 2D (call after EndMode3D) ──────────────────────
    void drawHoverTooltip(Camera3D& cam) {
        if (hoveredLayer < 0) return;

        const char* activationName = "—";
        if (hoveredLayer == 0)
            activationName = "Input";
        else if (hoveredLayer >= 1 && hoveredLayer <= 5)
            activationName = "ReLU";
        else if (hoveredLayer == 7)
            activationName = "Softmax";
        else if (hoveredLayer == 6)
            activationName = "Linear";

        // Format lines
        char line1[128], line2[128], line3[128];
        snprintf(line1, sizeof(line1), "%s", layerInfos[hoveredLayer].name);

        // Fix #2: show Pos for 2D grids (input + conv), show Node index for 1D (MLP)
        bool is2D = (tvs[hoveredLayer]->tW > 1 && tvs[hoveredLayer]->tH > 1);
        if (is2D && tvs[hoveredLayer]->tC > 1) {
            snprintf(line2, sizeof(line2), "Channel: %d  Pos: (%d, %d)", hoveredZ, hoveredX, hoveredY);
        } else if (is2D) {
            snprintf(line2, sizeof(line2), "Pos: (%d, %d)", hoveredX, hoveredY);
        } else {
            snprintf(line2, sizeof(line2), "Index: %d", hoveredX);
        }

        // Fix #8: only show value if animation has computed it
        if (hoveredComputed) {
            snprintf(line3, sizeof(line3), "Value: %.4f  (%s)", hoveredValue, activationName);
        } else {
            snprintf(line3, sizeof(line3), "Value: —  (not yet computed)");
        }

        int fontSize = 16;
        int pad = 10;
        int lineH = fontSize + 4;
        int w1 = MeasureText(line1, fontSize);
        int w2 = MeasureText(line2, fontSize);
        int w3 = MeasureText(line3, fontSize);
        int boxW = std::max({w1, w2, w3}) + pad * 2;
        int boxH = lineH * 3 + pad * 2;

        int tx = (int)hoverScreenPos.x + 20;
        int ty = (int)hoverScreenPos.y - boxH / 2;

        int screenW = GetScreenWidth();
        int screenH = GetScreenHeight();
        if (tx + boxW > screenW - 10) tx = (int)hoverScreenPos.x - boxW - 20;
        if (ty < 10) ty = 10;
        if (ty + boxH > screenH - 10) ty = screenH - boxH - 10;

        DrawRectangle(tx, ty, boxW, boxH, Fade(BLACK, 0.85f));
        DrawRectangleLines(tx, ty, boxW, boxH, layerInfos[hoveredLayer].headerColor);

        DrawText(line1, tx + pad, ty + pad, fontSize, layerInfos[hoveredLayer].headerColor);
        DrawText(line2, tx + pad, ty + pad + lineH, fontSize, LIGHTGRAY);
        DrawText(line3, tx + pad, ty + pad + lineH * 2, fontSize, WHITE);
    }

    // ─── Draw the layer info panel in 2D (call after EndMode3D) ───────────────
    void drawLayerInfoPanel() {
        if (selectedLayer < 0) return;

        const LayerInfo& info = layerInfos[selectedLayer];

        int fontSize = 16;
        int titleFontSize = 22;
        int pad = 16;
        int lineH = fontSize + 6;

        std::vector<std::string> descLines;
        {
            std::string desc(info.description);
            size_t pos = 0;
            while ((pos = desc.find('\n')) != std::string::npos) {
                descLines.push_back(desc.substr(0, pos));
                desc.erase(0, pos + 1);
            }
            if (!desc.empty()) descLines.push_back(desc);
        }

        int maxTextW = MeasureText(info.name, titleFontSize);
        char opLine[128], shapeLine[128];
        snprintf(opLine, sizeof(opLine), "Operation:  %s", info.operation);
        snprintf(shapeLine, sizeof(shapeLine), "Shape:      %s", info.shape);
        maxTextW = std::max(maxTextW, MeasureText(opLine, fontSize));
        maxTextW = std::max(maxTextW, MeasureText(shapeLine, fontSize));
        for (auto& dl : descLines) {
            maxTextW = std::max(maxTextW, MeasureText(dl.c_str(), fontSize));
        }

        int panelW = maxTextW + pad * 2 + 10;
        int headerH = titleFontSize + pad * 2;
        int bodyH = lineH * (3 + (int)descLines.size() + 1) + pad * 2;
        int panelH = headerH + bodyH;

        int screenW = GetScreenWidth();
        int screenH = GetScreenHeight();
        int px = screenW - panelW - 20;
        int py = (screenH - panelH) / 2;

        DrawRectangle(px, py, panelW, panelH, Fade(BLACK, 0.92f));

        DrawRectangle(px, py, panelW, headerH, Fade(info.headerColor, 0.9f));
        DrawText(info.name, px + pad, py + pad, titleFontSize, BLACK);

        DrawLine(px + pad, py + headerH + 4, px + panelW - pad, py + headerH + 4,
                 Fade(info.headerColor, 0.5f));

        int cy = py + headerH + pad;
        DrawText(opLine, px + pad, cy, fontSize, WHITE);
        cy += lineH;
        DrawText(shapeLine, px + pad, cy, fontSize, WHITE);
        cy += lineH + lineH / 2;

        for (auto& dl : descLines) {
            DrawText(dl.c_str(), px + pad, cy, fontSize, LIGHTGRAY);
            cy += lineH;
        }

        cy += lineH / 2;
        // Fix #5: just say "Click again to close"
        DrawText("[Click again to close]", px + pad, cy, 14, Fade(WHITE, 0.5f));

        DrawRectangleLines(px, py, panelW, panelH, Fade(info.headerColor, 0.6f));
    }
};
