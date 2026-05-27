// PlateRecognizer.cpp
// Stage 2: plate BGR crops → recognised strings.
// Two GPU waves across P plate streams:
//   Wave 1 (parallel): preprocess → resize → Otsu → CCL+filter → managed m_filtered
//   Wave 2 (parallel): charROIResize → KNN → async D2H
// Between waves, CPU groups char blobs (fast, O(N) per plate).

#include "PlateRecognizer.h"
#include "Preprocess.cuh"
#include "CCL.cuh"
#include "ImageOps.cuh"
#include "OtsuThreshold.cuh"
#include "KNN.cuh"
#include "CharGrouping.h"

#include <algorithm>

extern KNNModel gKnnModel;

// =============================================================================
void PlateRecognizer::recognizePlates(std::vector<PossiblePlate>& plates,
                                       PipelineContext&             ctx)
{
    if (plates.empty()) return;

    // ── WAVE 1: preprocessing + CCL+filter on each plate's stream ─────────────
    for (auto& plate : plates) {
        const int slot = plate.poolSlot;
        if (slot < 0) continue;

        PlateBuffer&  pb     = ctx.plateBuffers[slot];
        cudaStream_t  stream = ctx.plateStreams[slot];
        const int     W      = plate.plateBgrW;
        const int     H      = plate.plateBgrH;
        const int     dstW   = (int)(W * 1.6f);
        const int     dstH   = (int)(H * 1.6f);

        preprocessDevice(pb.d_plate_bgr, pb.d_thresh, W, H, pb.preproc, stream);
        runResizeInto(pb.d_thresh, W, H, pb.d_thresh_big, dstW, dstH, stream);
        runOtsuThreshold(pb.d_thresh_big, pb.d_thresh_otsu, dstW, dstH, stream);

        // CCL + filter → async D2H to pinned pb.h_filtered
        runCCLWithFilter(pb.d_thresh_otsu, dstW, dstH,
                         pb.plateWS,
                         pb.d_filtered, pb.d_num_filtered,
                         pb.h_filtered, pb.h_num_filtered,
                         stream);
    }

    // Wait for all plate streams to complete Wave 1.
    cudaDeviceSynchronize();

    // ── CPU: group chars per plate ─────────────────────────────────────────────
    struct PlateWork {
        int                       slot;
        int                       dstW, dstH;
        std::vector<PossibleChar> chars;
        std::vector<Rect2i>       rects;
    };
    std::vector<PlateWork> work;
    work.reserve(plates.size());

    for (const auto& plate : plates) {
        const int slot = plate.poolSlot;
        if (slot < 0) continue;

        PlateBuffer& pb   = ctx.plateBuffers[slot];
        int numFilt       = *pb.h_num_filtered;
        auto best         = bestCharGroup(pb.h_filtered, numFilt);
        if (best.empty()) continue;

        PlateWork pw;
        pw.slot  = slot;
        pw.dstW  = (int)(plate.plateBgrW * 1.6f);
        pw.dstH  = (int)(plate.plateBgrH * 1.6f);
        pw.chars = best;
        pw.rects.reserve(best.size());
        for (const auto& c : best) pw.rects.push_back(c.boundingRect);
        work.push_back(std::move(pw));
    }

    // ── WAVE 2: charROIResize + KNN on each plate's stream ───────────────────
    for (auto& pw : work) {
        PlateBuffer& pb     = ctx.plateBuffers[pw.slot];
        cudaStream_t stream = ctx.plateStreams[pw.slot];
        int N               = (int)std::min((int)pw.chars.size(), MAX_CHARS_PER_PLATE);

        runCharROIResize(pb.d_thresh_otsu, pw.dstW, pw.dstH,
                         pw.rects.data(), N,
                         pb.d_rects, pb.d_queries, stream);

        runKNNDevice(gKnnModel, pb.d_queries, N,
                     pb.d_knn_results, pb.h_labels, stream);
    }

    cudaDeviceSynchronize();

    // ── CPU: assemble plate strings ────────────────────────────────────────────
    for (const auto& pw : work) {
        PlateBuffer& pb = ctx.plateBuffers[pw.slot];
        int N           = (int)std::min((int)pw.chars.size(), MAX_CHARS_PER_PLATE);
        for (auto& plate : plates) {
            if (plate.poolSlot == pw.slot) {
                plate.strChars = assembleString(pb.h_labels, N);
                break;
            }
        }
    }
}

// =============================================================================
std::vector<PossibleChar> PlateRecognizer::bestCharGroup(
    const FilteredBlob* blobs, int count)
{
    if (count == 0) return {};
    auto chars  = blobsToChars(blobs, count);
    auto groups = groupChars(chars);
    if (groups.empty()) return {};

    for (auto& g : groups) {
        std::sort(g.begin(), g.end(), PossibleChar::sortCharsLeftToRight);
        g = removeInnerOverlapping(g);
    }
    return *std::max_element(groups.begin(), groups.end(),
        [](const auto& a, const auto& b){ return a.size() < b.size(); });
}

// =============================================================================
std::string PlateRecognizer::assembleString(const int32_t* labels, int count)
{
    std::string s;
    s.reserve(count);
    for (int i = 0; i < count; i++) s += static_cast<char>(labels[i]);
    return s;
}
