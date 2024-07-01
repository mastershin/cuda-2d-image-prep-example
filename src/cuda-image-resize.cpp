#include <filesystem>
#include <iostream>
#include <npp.h>
#include <opencv2/opencv.hpp>
#include <vector>

#include <nppi.h>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define NPP_CHECK(call)                                                        \
  do {                                                                         \
    NppStatus status = call;                                                   \
    if (status != NPP_SUCCESS) {                                               \
      std::cerr << "NPP error: " << status << " at " << __FILE__ << ":"        \
                << __LINE__ << std::endl;                                      \
      return;                                                                  \
    }                                                                          \
  } while (0)

// Function to parse command line arguments
void parseArguments(int argc, char **argv, std::string &inputDir,
                    std::string &outputDir, int &width, int &height) {
  width = 32;
  height = 32;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--directory" && i + 1 < argc) {
      inputDir = argv[++i];
    } else if (std::string(argv[i]) == "--output" && i + 1 < argc) {
      outputDir = argv[++i];
    }
    else if (std::string(argv[i]) == "--width" && i + 1 < argc) {
      width = std::stoi(argv[++i]);
    }
    else if (std::string(argv[i]) == "--height" && i + 1 < argc) {
      height = std::stoi(argv[++i]);
    }
  }
  if (inputDir.empty()) {
    throw std::invalid_argument(
        "Both --directory and --output must be specified");
  }
  if (outputDir.empty()) {
    std::cout << "--output directory not specified, which will only calculate "
                 "mean RGB"
              << std::endl;
  }

//   if (width <= 0 || height <= 0) {
//     throw std::invalid_argument("Invalid width or height");
//   }
}

// Function to calculate mean RGB values using NPP
void calculateMeanRGB(const std::vector<cv::Mat> &images, cv::Vec3f &meanRGB) {

  double sumR = 0.0, sumG = 0.0, sumB = 0.0;
  int totalPixels = 0;

  for (const auto &img : images) {
    int pixels = img.rows * img.cols;
    int channels = img.channels();
    totalPixels += pixels;

    NppiSize oSizeROI = {img.cols, img.rows};
    // Npp64f pSum[3] = {0.0, 0.0, 0.0};

    // Npp64f *pSum = new Npp64f[3]{0.0, 0.0, 0.0}; // Allocate on heap

    Npp64f *pSum;
    cudaMallocManaged(&pSum, 3 * sizeof(Npp64f));
    pSum[0] = pSum[1] = pSum[2] = 0.0;

    // Allocate device memory for the image
    // int stepBytes;
    // Npp8u* d_img = nppiMalloc_8u_C3(img.step, img.rows, &stepBytes);

    Npp8u *d_img;
    CUDA_CHECK(cudaMalloc(&d_img, pixels * channels * sizeof(Npp8u)));

    // link error?
    // status = nppiCopy_8u_C3R(img.data, img.step, d_img, img.step, oSizeROI);
    // if (NPP_NO_ERROR != status) {
    //     std::cerr << "Sum - Npp status: " << status << std::endl;
    //     return;
    // }

    CUDA_CHECK(cudaMemcpy(d_img, img.data, pixels * channels * sizeof(Npp8u),
                          cudaMemcpyHostToDevice));

    // Allocate device buffer

    int bufferSize;
    NPP_CHECK(nppiSumGetBufferHostSize_8u_C3R(oSizeROI, &bufferSize));
    CUDA_CHECK(cudaDeviceSynchronize());

    Npp8u *pDeviceBuffer;
    CUDA_CHECK(cudaMalloc(&pDeviceBuffer, bufferSize));

    NPP_CHECK(nppiSum_8u_C3R(d_img, img.step, oSizeROI, pDeviceBuffer, pSum));

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the sum results from the device buffer to the host
    // since we're using Managed Memory, no need to copy the buffer back.
    // cudaMemcpy(pSum, pDeviceBuffer, sizeof(Npp64f) * 3,
    // cudaMemcpyDeviceToHost);

    sumR += pSum[0];
    sumG += pSum[1];
    sumB += pSum[2];

    std::cout << "Image: " << img.rows << "x" << img.cols
              << ", Mean RGB: R=" << (pSum[0] / pixels)
              << ", G=" << pSum[1] / pixels << ", B=" << pSum[2] / pixels << " "
              << img.at<cv::Vec3b>(50, 50) << std::endl;

    // Free device memory
    // nppiFree(d_img);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_img));
    CUDA_CHECK(cudaFree(pDeviceBuffer));
    CUDA_CHECK(cudaFree(pSum));
    // delete[] pSum;
  }

  meanRGB[0] = static_cast<float>(sumR / totalPixels);
  meanRGB[1] = static_cast<float>(sumG / totalPixels);
  meanRGB[2] = static_cast<float>(sumB / totalPixels);
}

// Function to process images: resize, subtract mean, and save
void processImages(const std::vector<cv::Mat> &images,
                   const std::vector<std::string> &filenames,
                   const std::string &outputDir, int width, int height,
                   const cv::Vec3f &meanRGB) {
  for (size_t i = 0; i < images.size(); ++i) {
    cv::Mat resizedImg, finalImg;
    cv::resize(images[i], resizedImg, cv::Size(width, height));

    // Subtract mean values
    resizedImg.convertTo(finalImg, CV_32FC3);

    finalImg -= cv::Scalar(meanRGB[2], meanRGB[1], meanRGB[0]);

    // Convert back to 8-bit image
    finalImg.convertTo(finalImg, CV_8UC3);

    // Save the processed image
    fs::path inputPath = filenames[i];
    fs::path outputPath = outputDir / inputPath.filename();
    outputPath.replace_extension(".png");
    cv::imwrite(outputPath.string(), finalImg);
  }
}

int main(int argc, char **argv) {
  std::string inputDir, outputDir;

  int width = 32;
  int height = 64;

  try {
    parseArguments(argc, argv, inputDir, outputDir, width, height);

    std::vector<cv::Mat> images;
    std::vector<std::string> filenames;
    for (const auto &entry : fs::directory_iterator(inputDir)) {
      if (entry.is_regular_file()) {
        cv::Mat img = cv::imread(entry.path().string());
        if (!img.empty()) {
          images.push_back(img);
          filenames.push_back(entry.path().string());
        }
      }
    }

    if (images.empty()) {
      std::cerr << "No images found in the specified directory." << std::endl;
      return 1;
    }

    cv::Vec3f meanRGB;
    calculateMeanRGB(images, meanRGB);
    std::cout << "Mean RGB values: R=" << meanRGB[0] << ", G=" << meanRGB[1]
              << ", B=" << meanRGB[2] << std::endl;

    if (!outputDir.empty()) {
      fs::create_directory(outputDir);
      processImages(images, filenames, outputDir, width, height, meanRGB);

      std::cout << "Processing complete. Processed images saved to "
                << outputDir << std::endl;
    } else {
      std::cout
          << "Only Mean RGB values calculated. No output directory specified, "
             "so processed images were not saved."
          << std::endl;
    }
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
    return 1;
  }

  return 0;
}
