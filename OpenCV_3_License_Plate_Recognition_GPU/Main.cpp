// Main.cpp

#include "Main.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>

namespace fs = std::filesystem;

struct BenchmarkSummary {
    int numImages = 0;
    int numDetectedPlates = 0;
    int numDetectedChars = 0;
    double totalTimeMs = 0.0;
    double latencyMsPerImage = 0.0;
    double throughputImgPerSec = 0.0;
};

static bool isSupportedImageExtension(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp";
}

static std::vector<std::string> collectImagePaths(const std::string& inputPath) {
    std::vector<std::string> imagePaths;
    fs::path path(inputPath);

    if (fs::exists(path)) {
        if (fs::is_regular_file(path)) {
            if (isSupportedImageExtension(path)) imagePaths.push_back(path.string());
            return imagePaths;
        }
        if (fs::is_directory(path)) {
            for (const auto& entry : fs::directory_iterator(path)) {
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
    for (const auto& match : globMatches) imagePaths.emplace_back(match);
    std::sort(imagePaths.begin(), imagePaths.end());
    return imagePaths;
}

static void writeSummaryCsv(const std::string& path, const BenchmarkSummary& summary, const std::string& platform) {
    std::ofstream out(path);
    out << "platform,num_images,total_time_ms,latency_ms_per_img,throughput_img_per_s,total_plates,total_chars\n";
    out << platform << ","
        << summary.numImages << ","
        << std::fixed << std::setprecision(4)
        << summary.totalTimeMs << ","
        << summary.latencyMsPerImage << ","
        << summary.throughputImgPerSec << ","
        << summary.numDetectedPlates << ","
        << summary.numDetectedChars << "\n";
}

///////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

    bool blnKNNTrainingSuccessful = loadKNNDataAndTrainKNN();           // attempt KNN training

    if (blnKNNTrainingSuccessful == false) {                            // if KNN training was not successful
                                                                        // show error message
        std::cout << std::endl << std::endl << "error: error: KNN traning was not successful" << std::endl << std::endl;
        return(0);                                                      // and exit program
    }

    std::vector<std::string> imagePaths;
    if (argc > 1) {
        imagePaths = collectImagePaths(argv[1]);
    } else {
        imagePaths = collectImagePaths("image*.png");
        if (imagePaths.empty()) imagePaths = collectImagePaths("../image*.png");
    }

    if (imagePaths.empty()) {
        std::cerr << "Error: No input images found." << std::endl;
        std::cerr << "Usage: " << argv[0] << " [image_file|images_dir|glob_pattern] [summary_csv]" << std::endl;
        return 1;
    }

    std::string summaryCsvPath = (argc > 2) ? argv[2] : "license_plate_gpu_benchmark.csv";

    BenchmarkSummary summary;
    int skippedImages = 0;

    auto totalStart = std::chrono::steady_clock::now();
    for (const auto& imagePath : imagePaths) {
        cv::Mat imgOriginalScene = cv::imread(imagePath);
        if (imgOriginalScene.empty()) {
            std::cerr << "Warning: Could not read image: " << imagePath << std::endl;
            skippedImages++;
            continue;
        }

        std::vector<PossiblePlate> vectorOfPossiblePlates = detectPlatesInScene(imgOriginalScene);
        vectorOfPossiblePlates = detectCharsInPlates(vectorOfPossiblePlates);

        summary.numImages++;
        summary.numDetectedPlates += static_cast<int>(vectorOfPossiblePlates.size());
        for (const auto& plate : vectorOfPossiblePlates) {
            summary.numDetectedChars += static_cast<int>(plate.strChars.length());
        }
    }
    auto totalEnd = std::chrono::steady_clock::now();

    summary.totalTimeMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
    if (summary.numImages > 0) {
        summary.latencyMsPerImage = summary.totalTimeMs / static_cast<double>(summary.numImages);
        summary.throughputImgPerSec = 1000.0 * static_cast<double>(summary.numImages) / summary.totalTimeMs;
    }

    std::cout << "\n=== GPU Benchmark Summary ===\n";
    std::cout << "Images processed: " << summary.numImages << "\n";
    std::cout << "Images skipped:   " << skippedImages << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Latency (ms/img): " << summary.latencyMsPerImage << "\n";
    std::cout << "Throughput (img/s): " << summary.throughputImgPerSec << "\n";
    std::cout << "Plates detected:  " << summary.numDetectedPlates << "\n";
    std::cout << "Chars recognized: " << summary.numDetectedChars << "\n";

    writeSummaryCsv(summaryCsvPath, summary, "GPU");
    std::cout << "Summary CSV: " << summaryCsvPath << std::endl;

    return (summary.numImages > 0) ? 0 : 1;
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

    int intFontFace = cv::FONT_HERSHEY_SIMPLEX;                             // choose a plain jane font
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


