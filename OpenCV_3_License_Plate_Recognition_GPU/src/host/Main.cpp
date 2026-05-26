// Main.cpp
// Thin controller: load KNN model, build PipelineContext, process images.

#include "SceneAnalyzer.h"
#include "PlateRecognizer.h"
#include "ImageIO.h"
#include "KNN.cuh"
#include <iostream>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

namespace fs = std::filesystem;

// Global KNN model — loaded once, used by PlateRecognizer via extern.
KNNModel gKnnModel;

struct BenchmarkSummary {
    int    numImages         = 0;
    int    numDetectedPlates = 0;
    int    numDetectedChars  = 0;
    double totalTimeMs       = 0.0;
};

static bool isSupportedExtension(const fs::path& p) {
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c){ return (char)std::tolower(c); });
    return ext == ".ppm";
}

static std::vector<std::string> collectImagePaths(const std::string& input) {
    std::vector<std::string> paths;
    fs::path p(input);
    if (fs::exists(p)) {
        if (fs::is_regular_file(p) && isSupportedExtension(p)) { paths.push_back(p.string()); return paths; }
        if (fs::is_directory(p))
            for (const auto& e : fs::directory_iterator(p))
                if (e.is_regular_file() && isSupportedExtension(e.path()))
                    paths.push_back(e.path().string());
    } else {
        for (const auto& e : fs::directory_iterator(fs::current_path()))
            if (e.is_regular_file() && isSupportedExtension(e.path()))
                if (e.path().filename().string().find("image") == 0)
                    paths.push_back(e.path().string());
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

static void writeCsv(const std::string& path, const BenchmarkSummary& s,
                     int maxPlates, const std::string& platform)
{
    std::ofstream out(path);
    out << "platform,max_plates,num_images,total_time_ms,latency_ms_per_img,"
           "throughput_img_per_s,total_plates,total_chars\n";
    double lat  = s.numImages > 0 ? s.totalTimeMs / s.numImages : 0.0;
    double tput = s.numImages > 0 ? 1000.0 * s.numImages / s.totalTimeMs : 0.0;
    out << platform << "," << maxPlates << "," << s.numImages << ","
        << std::fixed << std::setprecision(4)
        << s.totalTimeMs << "," << lat << "," << tput << ","
        << s.numDetectedPlates << "," << s.numDetectedChars << "\n";
}

// =============================================================================
// Loader thread: reads .ppm files from disk into a bounded queue so the main
// thread can upload to the GPU without waiting on disk I/O.
// =============================================================================
struct LoaderQueue {
    std::deque<HostImage>   items;
    std::mutex              mtx;
    std::condition_variable ready;
    std::condition_variable space;
    bool                    done = false;
    static constexpr int    CAPACITY = 3;
};

static void runLoader(const std::vector<std::string>& paths, LoaderQueue& q)
{
    for (const auto& p : paths) {
        HostImage img = loadPPM(p);
        std::unique_lock<std::mutex> lk(q.mtx);
        q.space.wait(lk, [&]{ return (int)q.items.size() < LoaderQueue::CAPACITY; });
        q.items.push_back(std::move(img));
        q.ready.notify_one();
    }
    {
        std::unique_lock<std::mutex> lk(q.mtx);
        q.done = true;
    }
    q.ready.notify_all();
}

// =============================================================================
int main(int argc, char** argv)
{
    // ── Parse arguments: [image_path] [--max-plates N] [csv_output] ──────────
    std::string imagePath, csvPath = "license_plate_gpu_benchmark.csv";
    int maxPlates = 8;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--max-plates" && i + 1 < argc) {
            maxPlates = std::max(1, std::atoi(argv[++i]));
        } else if (arg.size() > 4 && arg.substr(arg.size()-4) == ".csv") {
            csvPath = arg;
        } else {
            imagePath = arg;
        }
    }

    // ── Load KNN model (GPU, persistent) ──────────────────────────────────────
    std::string knnPath = (fs::canonical("/proc/self/exe").parent_path() / "knn_data.bin").string();
    if (!loadKNNModel(knnPath.c_str(), gKnnModel)) {
        std::cerr << "Error: knn_data.bin not found next to binary (" << knnPath << ")\n";
        return 1;
    }

    // ── Collect images ─────────────────────────────────────────────────────────
    auto paths = imagePath.empty() ? collectImagePaths(".") : collectImagePaths(imagePath);
    if (paths.empty() && imagePath.empty()) paths = collectImagePaths("..");
    if (paths.empty()) {
        std::cerr << "Error: no .ppm images found.\n"
                  << "Usage: " << argv[0] << " [image.ppm|dir] [--max-plates N] [out.csv]\n";
        return 1;
    }

    // ── Build pipeline context once (allocates pool + streams) ────────────────
    PipelineContext ctx = PipelineContext::create(maxPlates);
    SceneAnalyzer   sceneAnalyzer;
    PlateRecognizer plateRecognizer;

    BenchmarkSummary summary;
    int skipped = 0;

    // Two host-side image slots — ping-pong so one can be uploading
    // while the other is being processed on the GPU.
    HostImage h_bufs[2];

    // Start loader thread — reads images from disk into a bounded queue.
    LoaderQueue queue;
    std::thread loader(runLoader, std::cref(paths), std::ref(queue));

    // Pop the next HostImage from the queue (blocks if empty, returns
    // a null HostImage when the queue is drained and the loader is done).
    auto popImage = [&]() -> HostImage {
        std::unique_lock<std::mutex> lk(queue.mtx);
        queue.ready.wait(lk, [&]{ return !queue.items.empty() || queue.done; });
        if (queue.items.empty()) return HostImage();
        HostImage img = std::move(queue.items.front());
        queue.items.pop_front();
        queue.space.notify_one();
        return img;
    };

    // Upload h_bufs[slot] to the GPU and record the completion event.
    // Always records the event so the compute stream never stalls on
    // a stale signal, even when the image failed to load.
    auto uploadToGPU = [&](int slot) {
        if (h_bufs[slot].data)
            cudaMemcpyAsync(ctx.sceneBuffers[slot].d_scene_bgr,
                            h_bufs[slot].data,
                            (size_t)SCENE_W * SCENE_H * 3,
                            cudaMemcpyHostToDevice, ctx.transferStream);
        cudaEventRecord(ctx.uploadDone[slot], ctx.transferStream);
    };

    // Pre-fill slot 0 before entering the loop.
    h_bufs[0] = popImage();
    uploadToGPU(0);

    auto totalStart = std::chrono::steady_clock::now();

    for (int i = 0; i < (int)paths.size(); i++) {
        const int cur = i % 2;
        const int nxt = (i + 1) % 2;

        cudaStreamWaitEvent(ctx.sceneStream, ctx.uploadDone[cur], 0);

        if (!h_bufs[cur].data) {
            skipped++;
            if (i + 1 < (int)paths.size()) {
                h_bufs[nxt] = popImage();
                uploadToGPU(nxt);
            }
            continue;
        }

        auto plates = sceneAnalyzer.detectPlates(ctx.sceneBuffers[cur], ctx);

        if (i + 1 < (int)paths.size()) {
            h_bufs[nxt] = popImage();
            uploadToGPU(nxt);
        }

        plateRecognizer.recognizePlates(plates, ctx);

        summary.numImages++;
        summary.numDetectedPlates += (int)plates.size();
        for (const auto& plate : plates) {
            summary.numDetectedChars += (int)plate.strChars.size();
            if (!plate.strChars.empty())
                std::cout << paths[i] << " -> " << plate.strChars << "\n";
        }
    }

    loader.join();

    auto totalEnd = std::chrono::steady_clock::now();
    summary.totalTimeMs =
        std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    double lat  = summary.numImages > 0 ? summary.totalTimeMs / summary.numImages : 0.0;
    double tput = summary.numImages > 0 ? 1000.0 * summary.numImages / summary.totalTimeMs : 0.0;

    std::cout << "\n=== GPU Benchmark Summary ===\n"
              << "Images processed : " << summary.numImages << "\n"
              << "Images skipped   : " << skipped << "\n"
              << "Max plates/image : " << maxPlates << "\n"
              << std::fixed << std::setprecision(3)
              << "Latency (ms/img) : " << lat  << "\n"
              << "Throughput (img/s): " << tput << "\n"
              << "Plates detected  : " << summary.numDetectedPlates << "\n"
              << "Chars recognised : " << summary.numDetectedChars  << "\n";

    writeCsv(csvPath, summary, maxPlates, "GPU-no-opencv");
    std::cout << "Summary CSV: " << csvPath << "\n";

    ctx.destroy();
    freeKNNModel(gKnnModel);
    return (summary.numImages > 0) ? 0 : 1;
}
