#ifndef TIMING_UTILS_H
#define TIMING_UTILS_H

#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <cmath>

// RAII Timer class for high-resolution timing
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    double elapsed_ms;
    bool stopped;

public:
    Timer() : elapsed_ms(0.0), stopped(false) {
        start_time = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        if (!stopped) stop();
    }

    double stop() {
        if (!stopped) {
            auto end_time = std::chrono::high_resolution_clock::now();
            elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            stopped = true;
        }
        return elapsed_ms;
    }

    double elapsed() const {
        if (stopped) return elapsed_ms;
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
};

// Per-image statistics structure
struct FrameStats {
    int frame_number;
    int image_width;
    int image_height;
    int num_plates_detected;
    int num_chars_recognized;

    double total_time_ms;
    double preprocess_time_ms;
    double find_plates_time_ms;
    double detect_chars_time_ms;

    double total_flops;
    double preprocess_flops;
    double find_plates_flops;
    double detect_chars_flops;

    double fps;
    double gflops;

    FrameStats() : frame_number(0), image_width(0), image_height(0),
                   num_plates_detected(0), num_chars_recognized(0),
                   total_time_ms(0.0), preprocess_time_ms(0.0),
                   find_plates_time_ms(0.0), detect_chars_time_ms(0.0),
                   total_flops(0.0), preprocess_flops(0.0),
                   find_plates_flops(0.0), detect_chars_flops(0.0),
                   fps(0.0), gflops(0.0) {}
};

// FLOP Counter helper class
class FLOPCounter {
private:
    double total_flops;

public:
    FLOPCounter() : total_flops(0.0) {}

    // Estimate FLOPs for findContours: O(n*log(n)) where n = image pixels
    void addFindContours(int image_width, int image_height) {
        long long n = (long long)image_width * image_height;
        total_flops += n * std::log2(n) * 2.0;  // approximate coefficient
    }

    // Estimate FLOPs for warpAffine: pixel_count * interpolation ops
    void addWarpAffine(int image_width, int image_height) {
        long long n = (long long)image_width * image_height;
        total_flops += n * 4.0;  // bilinear interpolation ≈ 4 ops per pixel
    }

    // Estimate FLOPs for getRectSubPix: sub-image operations
    void addGetRectSubPix(int rect_width, int rect_height) {
        long long n = (long long)rect_width * rect_height;
        total_flops += n * 2.0;  // interpolation + copy
    }

    // Estimate FLOPs for KNN matching: k-nearest with Euclidean distance
    // distance = sqrt(sum((a-b)^2)) for each training sample
    void addKNNMatching(int num_query_samples, int num_training_samples, int feature_dim) {
        // Each query: num_training * (feature_dim multiplications + feature_dim subtractions + 1 sqrt)
        total_flops += (double)num_query_samples * num_training_samples * (feature_dim * 2.0 + 10.0);
    }

    // Estimate FLOPs for image preprocessing (grayscale + threshold)
    void addPreprocess(int image_width, int image_height) {
        long long n = (long long)image_width * image_height;
        total_flops += n * 3.0;  // grayscale conversion + threshold
    }

    double getTotalFLOPs() const { return total_flops; }
    void reset() { total_flops = 0.0; }
};

// Performance reporter
class PerformanceReporter {
private:
    std::vector<FrameStats> frame_stats;
    std::string csv_filename;

public:
    PerformanceReporter(const std::string &filename = "license_plate_cpu_baseline.csv")
        : csv_filename(filename) {}

    void addFrameStats(const FrameStats &stats) {
        frame_stats.push_back(stats);
    }

    void printSummary() const {
        if (frame_stats.empty()) {
                std::cout << "No image statistics recorded." << std::endl;
            return;
        }

        double avg_fps = 0.0, avg_gflops = 0.0;
        double avg_total_time = 0.0, avg_find_plates = 0.0, avg_detect_chars = 0.0;
        int total_plates = 0, total_chars = 0;

        for (const auto &stats : frame_stats) {
            avg_fps += stats.fps;
            avg_gflops += stats.gflops;
            avg_total_time += stats.total_time_ms;
            avg_find_plates += stats.find_plates_time_ms;
            avg_detect_chars += stats.detect_chars_time_ms;
            total_plates += stats.num_plates_detected;
            total_chars += stats.num_chars_recognized;
        }

        int n = frame_stats.size();
        avg_fps /= n;
        avg_gflops /= n;
        avg_total_time /= n;
        avg_find_plates /= n;
        avg_detect_chars /= n;

        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "CPU PERFORMANCE BASELINE SUMMARY" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "Images processed: " << n << std::endl;
        std::cout << "Average FPS: " << std::fixed << std::setprecision(2) << avg_fps << std::endl;
        std::cout << "Average GFLOPS: " << std::fixed << std::setprecision(2) << avg_gflops << std::endl;
        std::cout << "Average total time per image: " << std::fixed << std::setprecision(2) << avg_total_time << " ms" << std::endl;
        std::cout << "\nTime breakdown per image:" << std::endl;
        std::cout << "  - Find plates: " << std::fixed << std::setprecision(2) << avg_find_plates << " ms" << std::endl;
        std::cout << "  - Detect chars: " << std::fixed << std::setprecision(2) << avg_detect_chars << " ms" << std::endl;
        std::cout << "\nTotal plates detected: " << total_plates << std::endl;
        std::cout << "Total characters recognized: " << total_chars << std::endl;
        std::cout << std::string(70, '=') << "\n" << std::endl;
    }

    void writeCSV() const {
        std::ofstream file(csv_filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open " << csv_filename << " for writing." << std::endl;
            return;
        }

        // Header
        file << "frame_number,width,height,num_plates,num_chars,"
             << "total_time_ms,find_plates_ms,detect_chars_ms,"
             << "total_flops,find_plates_flops,detect_chars_flops,"
             << "fps,gflops\n";

        // Data rows
        for (const auto &stats : frame_stats) {
            file << stats.frame_number << ","
                 << stats.image_width << "," << stats.image_height << ","
                 << stats.num_plates_detected << "," << stats.num_chars_recognized << ","
                 << std::fixed << std::setprecision(3)
                 << stats.total_time_ms << ","
                 << stats.find_plates_time_ms << ","
                 << stats.detect_chars_time_ms << ","
                 << stats.total_flops << ","
                 << stats.find_plates_flops << ","
                 << stats.detect_chars_flops << ","
                 << stats.fps << ","
                 << stats.gflops << "\n";
        }

        file.close();
        std::cout << "CSV report written to: " << csv_filename << std::endl;
    }
};

#endif // TIMING_UTILS_H

