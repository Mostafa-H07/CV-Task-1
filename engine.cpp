#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
#include <random>

using namespace cv;
using namespace std;

// --- OOP IMAGE PROCESSOR CLASS ---
class ImageProcessor {
private:
    Mat img;

    Mat makeZeroPadded(int top, int bottom, int left, int right) const {
        Mat padded = Mat::zeros(
            img.rows + top + bottom,
            img.cols + left + right,
            img.type()
        );

        for (int y = 0; y < img.rows; ++y) {
            const uchar* source = img.ptr<uchar>(y);
            uchar* destination = padded.ptr<uchar>(y + top) + left;
            memcpy(destination, source, static_cast<size_t>(img.cols));
        }
        return padded;
    }

public:
    // Constructor maps Python's flat 1D array into a 2D OpenCV Matrix
    ImageProcessor(uchar* data, int rows, int cols) {
        img = Mat(rows, cols, CV_8U, data).clone();
    }

    // --- NOISE GENERATION ---
    void addUniformNoise(int lower, int upper) {
        Mat noise(img.size(), CV_16S);
        randu(noise, lower, upper);
        
        Mat temp;
        img.convertTo(temp, CV_16S);
        temp += noise;
        
        temp = cv::max(temp, 0);
        temp = cv::min(temp, 255);
        temp.convertTo(img, CV_8U);
    }

    void addGaussianNoise(double mean, double stddev) {
        Mat noise(img.size(), CV_16S);
        randn(noise, mean, stddev);
        
        Mat temp;
        img.convertTo(temp, CV_16S);
        temp += noise;
        
        temp = cv::max(temp, 0);
        temp = cv::min(temp, 255);
        temp.convertTo(img, CV_8U);
    }

    void addSaltAndPepper(double ratio, double density) {
        double s = density / (1.0 + ratio);
        double p = s * ratio;
        mt19937 gen(1337); 
        uniform_real_distribution<double> dist(0.0, 1.0);

        for (int y = 0; y < img.rows; y++) {
            for (int x = 0; x < img.cols; x++) {
                double prob = dist(gen);
                if (prob < p) img.at<uchar>(y, x) = 0;
                else if (prob > 1.0 - s) img.at<uchar>(y, x) = 255;
            }
        }
    }

    // --- FILTER FACTORY & CONVOLUTION ---
    Mat generateKernel(int type, int size, int dir, double mean, double stddev) {
        if (type == 1) return Mat::ones(size, size, CV_32F) / (float)(size * size); // Avg
        if (type == 2) { // Gaussian
            if (stddev <= 0.0) stddev = 1.0;
            Mat kernel(size, size, CV_32F);
            float sum = 0.0;
            int half = size / 2;
            for (int y = -half; y <= half; y++) {
                for (int x = -half; x <= half; x++) {
                    double dx = x - mean;
                    double dy = y - mean;
                    float val = static_cast<float>(exp(-(dx * dx + dy * dy) /
                        (2.0 * stddev * stddev)));
                    kernel.at<float>(y + half, x + half) = val;
                    sum += val;
                }
            }
            return kernel / sum;
        }
        if (type == 4) { // Sobel
            float s_data[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
            Mat k = Mat(3, 3, CV_32F, s_data).clone();
            return (dir == 1) ? k.t() : k;
        }
        if (type == 5) { // Prewitt
            float p_data[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
            Mat k = Mat(3, 3, CV_32F, p_data).clone();
            return (dir == 1) ? k.t() : k;
        }
        if (type == 6) { // Roberts
            if (dir == 0) { float rx[4] = {1, 0, 0, -1}; return Mat(2, 2, CV_32F, rx).clone(); }
            else          { float ry[4] = {0, 1, -1, 0}; return Mat(2, 2, CV_32F, ry).clone(); }
        }
        return Mat();
    }

    void applyConvolution(const Mat& kernel) {
        int pad_h = kernel.rows / 2;
        int pad_w = kernel.cols / 2;
        Mat padded = makeZeroPadded(pad_h, pad_h, pad_w, pad_w);
        
        Mat output = Mat::zeros(img.size(), CV_32F);
        
        for (int y = 0; y < img.rows; ++y) {
            for (int x = 0; x < img.cols; ++x) {
                float sum = 0.0;
                for (int ky = 0; ky < kernel.rows; ++ky) {
                    const uchar* imageRow = padded.ptr<uchar>(y + ky);
                    const float* kernelRow = kernel.ptr<float>(ky);
                    for (int kx = 0; kx < kernel.cols; ++kx) {
                        sum += imageRow[x + kx] * kernelRow[kx];
                    }
                }
                output.at<float>(y, x) = sum;
            }
        }
        output = cv::abs(output);
        output.convertTo(img, CV_8U);
    }

    void applyMedian(int size) {
        int pad = size / 2;
        Mat padded = makeZeroPadded(pad, pad, pad, pad);
        Mat output(img.size(), CV_8U);
        vector<uchar> window(size * size);
        
        for (int y = 0; y < img.rows; ++y) {
            for (int x = 0; x < img.cols; ++x) {
                int k = 0;
                for (int ky = 0; ky < size; ++ky) {
                    for (int kx = 0; kx < size; ++kx) {
                        window[k++] = padded.at<uchar>(y + ky, x + kx);
                    }
                }
                nth_element(window.begin(), window.begin() + window.size() / 2, window.end());
                output.at<uchar>(y, x) = window[window.size() / 2];
            }
        }
        img = output;
    }

    // Histogram equalization: new = round(255 * CDF[old])
    void applyEqualize() {
        int hist[256] = {0};
        for (int y = 0; y < img.rows; y++)
            for (int x = 0; x < img.cols; x++)
                hist[img.at<uchar>(y, x)]++;

        double total = (double)img.rows * img.cols;
        double cdf = 0.0;
        uchar lut[256];
        for (int i = 0; i < 256; i++) {
            cdf += hist[i] / total;
            lut[i] = saturate_cast<uchar>(cvRound(255.0 * cdf));
        }
        for (int y = 0; y < img.rows; y++)
            for (int x = 0; x < img.cols; x++)
                img.at<uchar>(y, x) = lut[img.at<uchar>(y, x)];
    }

    // Normalization: linear stretch to [0, 255]
    void applyNormalize() {
        double mn, mx;
        minMaxLoc(img, &mn, &mx);
        if (mx == mn) return;
        double scale = 255.0 / (mx - mn);
        img.convertTo(img, CV_8U, scale, -mn * scale);
    }

    void applyCanny() {
        Mat edges;
        Canny(img, edges, 100, 200); // Per task 1, OpenCV is explicitly used here
        img = edges;
    }

    // --- DATA EXTRACTION ---
    void copyHistogramTo(int* out_hist_data) {
        int hist[256] = {0};

        for (int y = 0; y < img.rows; y++) {
            for (int x = 0; x < img.cols; x++) {
                int val = img.at<uchar>(y, x);
                hist[val]++;
            }
        }
        memcpy(out_hist_data, hist, sizeof(hist));
    }

    void copyImageTo(uchar* out_data) {
        memcpy(out_data, img.data, img.total());
    }
};

// --- C-LINKAGE WRAPPER FOR PYTHON ---
extern "C" {
    void run_pipeline(unsigned char* in_data, int rows, int cols,
                      int noise_type, float np1, float np2,
                      int filter_type, int k_size, int dir,
                      float filter_mean, float filter_stddev,
                      unsigned char* out_noisy, unsigned char* out_img,
                      int* out_hist) {
        
        // 1. Initialize OOP Object
        ImageProcessor proc(in_data, rows, cols);

        // 2. Add Noise
        if (noise_type == 1) proc.addUniformNoise((int)np1, (int)np2);
        if (noise_type == 2) proc.addGaussianNoise(np1, np2);
        if (noise_type == 3) proc.addSaltAndPepper(np1, np2);

        proc.copyImageTo(out_noisy);

        // 3. Apply Filters & Edge Detection
        if (filter_type == 3) {
            proc.applyMedian(k_size);
        } else if (filter_type == 7) {
            proc.applyCanny();
        } else if (filter_type == 8) {
            proc.applyEqualize();
        } else if (filter_type == 9) {
            proc.applyNormalize();
        } else if (filter_type > 0) {
            Mat k = proc.generateKernel(filter_type, k_size, dir, filter_mean, filter_stddev);
            proc.applyConvolution(k);
        }

        // 4. Output Data
        proc.copyImageTo(out_img);
        proc.copyHistogramTo(out_hist);
        
        
    }
}