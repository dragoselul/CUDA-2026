// Main.cpp

#include "Main.h"
#include "timing_utils.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

static bool isSupportedImageExtension(const fs::path &path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return (char) std::tolower(c); });
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp";
}

static std::vector<std::string> collectImagePaths(const std::string &inputPath) {
    std::vector<std::string> imagePaths;

    fs::path path(inputPath);
    if (fs::exists(path)) {
        if (fs::is_regular_file(path)) {
            if (isSupportedImageExtension(path)) {
                imagePaths.push_back(path.string());
            }
            return imagePaths;
        }

        if (fs::is_directory(path)) {
            for (const auto &entry : fs::directory_iterator(path)) {
                if (entry.is_regular_file() && isSupportedImageExtension(entry.path())) {
                    imagePaths.push_back(entry.path().string());
                }
            }
            std::sort(imagePaths.begin(), imagePaths.end());
            return imagePaths;
        }
    }

    std::vector<cv::String> globMatches;
    cv::glob(inputPath, globMatches, false);
    for (const auto &match : globMatches) {
        imagePaths.emplace_back(match);
    }
    std::sort(imagePaths.begin(), imagePaths.end());

    return imagePaths;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

    std::string dataDir = fs::canonical("/proc/self/exe").parent_path().string();
    bool blnKNNTrainingSuccessful = loadKNNDataAndTrainKNN(dataDir);

    if (blnKNNTrainingSuccessful == false) {
        std::cout << std::endl << std::endl << "error: KNN training was not successful" << std::endl << std::endl;
        return(0);
    }

    std::vector<std::string> imagePaths;
    if (argc > 1) {
        imagePaths = collectImagePaths(argv[1]);
    } else {
        imagePaths = collectImagePaths("image*.png");
        if (imagePaths.empty()) {
            imagePaths = collectImagePaths("../image*.png");
        }
    }

    if (imagePaths.empty()) {
        std::cerr << "Error: No input images found." << std::endl;
        std::cerr << "Usage: " << argv[0] << " [image_file|images_dir|glob_pattern] [csv_output]" << std::endl;
        std::cerr << "Example: " << argv[0] << " ../image*.png" << std::endl;
        std::cerr << "Example: " << argv[0] << " ../" << std::endl;
        return(1);
    }

    std::string csvOutput = (argc > 2) ? argv[2] : "license_plate_cpu_baseline.csv";
    std::string summaryCsvOutput = (argc > 3) ? argv[3] : "license_plate_cpu_summary.csv";

    std::cout << "Running CPU image benchmark on " << imagePaths.size() << " image(s)" << std::endl;
    std::cout << "Execution mode: CPU (no cv::cuda path in this target)" << std::endl;
    std::cout << "Starting image processing..." << std::endl;

    // Performance reporting
    PerformanceReporter reporter(csvOutput);

    int image_number = 0;
    int skipped_images = 0;
    double benchmarkTotalTimeMs = 0.0;
    int benchmarkTotalPlates = 0;
    int benchmarkTotalChars = 0;

    // Main processing loop over images
    for (const auto &imagePath : imagePaths) {
        cv::Mat frame = cv::imread(imagePath);
        if (frame.empty()) {
            std::cerr << "Warning: Could not read image: " << imagePath << std::endl;
            skipped_images++;
            continue;
        }
        image_number++;

        // Image processing statistics
        FrameStats stats;
        stats.frame_number = image_number;
        stats.image_width = frame.cols;
        stats.image_height = frame.rows;

        // Overall frame timer
        Timer frame_timer;

        // Detect plates
        Timer detect_plates_timer;
        std::vector<PossiblePlate> vectorOfPossiblePlates = detectPlatesInScene(frame);
        stats.find_plates_time_ms = detect_plates_timer.stop();
        stats.num_plates_detected = vectorOfPossiblePlates.size();

        // Detect characters in plates
        Timer detect_chars_timer;
        vectorOfPossiblePlates = detectCharsInPlates(vectorOfPossiblePlates);
        stats.detect_chars_time_ms = detect_chars_timer.stop();

        // Count total characters recognized
        for (const auto &plate : vectorOfPossiblePlates) {
            stats.num_chars_recognized += plate.strChars.length();
        }

        stats.total_time_ms = frame_timer.stop();

        // Estimate FLOPs
        FLOPCounter flop_counter;
        flop_counter.addFindContours(frame.cols, frame.rows);
        flop_counter.addWarpAffine(frame.cols, frame.rows);
        for (const auto &plate : vectorOfPossiblePlates) {
            flop_counter.addGetRectSubPix(plate.imgPlate.cols, plate.imgPlate.rows);
        }
        flop_counter.addKNNMatching(stats.num_chars_recognized, 1000, 784);  // Rough estimate: 1000 training samples

        stats.total_flops = flop_counter.getTotalFLOPs();
        stats.find_plates_flops = stats.total_flops * 0.6;  // ~60% in plate detection
        stats.detect_chars_flops = stats.total_flops * 0.4;  // ~40% in char recognition

        // Calculate performance metrics
        if (stats.total_time_ms > 0.0) {
            stats.fps = 1000.0 / stats.total_time_ms;
            stats.gflops = (stats.total_flops / 1e9) / (stats.total_time_ms / 1000.0);
        }

        // Record statistics
        reporter.addFrameStats(stats);

        benchmarkTotalTimeMs += stats.total_time_ms;
        benchmarkTotalPlates += stats.num_plates_detected;
        benchmarkTotalChars += stats.num_chars_recognized;

        // Print per-image progress
        std::cout << "Image " << image_number << " (" << imagePath << "): "
                      << std::fixed << std::setprecision(2)
                      << stats.fps << " FPS, "
                      << stats.gflops << " GFLOPS, "
                      << stats.num_plates_detected << " plates, "
                      << stats.num_chars_recognized << " chars" << std::endl;
    }

    cv::destroyAllWindows();

    std::cout << "\nProcessing complete!" << std::endl;
    std::cout << "Total images processed: " << image_number << std::endl;
    if (skipped_images > 0) {
        std::cout << "Skipped unreadable images: " << skipped_images << std::endl;
    }

    if (image_number == 0) {
        std::cerr << "Error: No valid images were processed." << std::endl;
        return(1);
    }

    // Print and save results
    reporter.printSummary();
    reporter.writeCSV();

    const double latencyMsPerImage = benchmarkTotalTimeMs / static_cast<double>(image_number);
    const double throughputImgPerSec = 1000.0 * static_cast<double>(image_number) / benchmarkTotalTimeMs;

    std::cout << "\n=== CPU Benchmark Summary ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Latency (ms/img): " << latencyMsPerImage << std::endl;
    std::cout << "Throughput (img/s): " << throughputImgPerSec << std::endl;
    std::cout << "Plates detected: " << benchmarkTotalPlates << std::endl;
    std::cout << "Chars recognized: " << benchmarkTotalChars << std::endl;

    std::ofstream summaryOut(summaryCsvOutput);
    summaryOut << "platform,num_images,total_time_ms,latency_ms_per_img,throughput_img_per_s,total_plates,total_chars\n";
    summaryOut << "CPU,"
               << image_number << ","
               << std::fixed << std::setprecision(4)
               << benchmarkTotalTimeMs << ","
               << latencyMsPerImage << ","
               << throughputImgPerSec << ","
               << benchmarkTotalPlates << ","
               << benchmarkTotalChars << "\n";
    std::cout << "Summary CSV: " << summaryCsvOutput << std::endl;

    return(0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawRedRectangleAroundPlate(cv::Mat &imgOriginalScene, PossiblePlate &licPlate) {
    cv::Point2f p2fRectPoints[4];

    licPlate.rrLocationOfPlateInScene.points(p2fRectPoints);            // get 4 vertices of rotated rect

    for (int i = 0; i < 4; i++) {                                       // draw 4 red lines
        cv::line(imgOriginalScene, p2fRectPoints[i], p2fRectPoints[(i + 1) % 4], SCALAR_RED, 2);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void writeLicensePlateCharsOnImage(cv::Mat &imgOriginalScene, PossiblePlate &licPlate) {
    cv::Point ptCenterOfTextArea;                   // this will be the center of the area the text will be written to
    cv::Point ptLowerLeftTextOrigin;                // this will be the bottom left of the area that the text will be written to

    int intFontFace = cv::FONT_HERSHEY_SIMPLEX;                              // choose a plain jane font
    double dblFontScale = (double)licPlate.imgPlate.rows / 30.0;            // base font scale on height of plate area
    int intFontThickness = (int)std::round(dblFontScale * 1.5);             // base font thickness on font scale
    int intBaseline = 0;

    cv::Size textSize = cv::getTextSize(licPlate.strChars, intFontFace, dblFontScale, intFontThickness, &intBaseline);      // call getTextSize

    ptCenterOfTextArea.x = (int)licPlate.rrLocationOfPlateInScene.center.x;         // the horizontal location of the text area is the same as the plate

    if (licPlate.rrLocationOfPlateInScene.center.y < (imgOriginalScene.rows * 0.75)) {      // if the license plate is in the upper 3/4 of the image
                                                                                            // write the chars in below the plate
        ptCenterOfTextArea.y = (int)std::round(licPlate.rrLocationOfPlateInScene.center.y) + (int)std::round((double)licPlate.imgPlate.rows * 1.6);
    }
    else {                                                                                // else if the license plate is in the lower 1/4 of the image
                                                                                          // write the chars in above the plate
        ptCenterOfTextArea.y = (int)std::round(licPlate.rrLocationOfPlateInScene.center.y) - (int)std::round((double)licPlate.imgPlate.rows * 1.6);
    }

    ptLowerLeftTextOrigin.x = (int)(ptCenterOfTextArea.x - (textSize.width / 2));           // calculate the lower left origin of the text area
    ptLowerLeftTextOrigin.y = (int)(ptCenterOfTextArea.y + (textSize.height / 2));          // based on the text area center, width, and height

                                                                                            // write the text on the image
    cv::putText(imgOriginalScene, licPlate.strChars, ptLowerLeftTextOrigin, intFontFace, dblFontScale, SCALAR_YELLOW, intFontThickness);
}


