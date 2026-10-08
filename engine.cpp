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
    Mat generateKernel(int type, int size, int dir) {
        if (type == 1) return Mat::ones(size, size, CV_32F) / (float)(size * size); // Avg
        if (type == 2) { // Gaussian
            Mat kernel(size, size, CV_32F);
            float sum = 0.0;
            int half = size / 2;
            for (int y = -half; y <= half; y++) {
                for (int x = -half; x <= half; x++) {
                    float val = exp(-(x * x + y * y) / 2.0f);
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
        Mat padded;
        copyMakeBorder(img, padded, pad_h, pad_h, pad_w, pad_w, BORDER_REPLICATE);
        
        Mat output = Mat::zeros(img.size(), CV_32F);
        
        for (int y = 0; y < img.rows; ++y) {
            for (int x = 0; x < img.cols; ++x) {
                float sum = 0.0;
                for (int ky = 0; ky < kernel.rows; ++ky) {
                    for (int kx = 0; kx < kernel.cols; ++kx) {
                        sum += padded.at<uchar>(y + ky, x + kx) * kernel.at<float>(ky, kx);
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
        Mat padded;
        copyMakeBorder(img, padded, pad, pad, pad, pad, BORDER_REPLICATE);
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

    void applyCanny() {
        Mat edges;
        Canny(img, edges, 100, 200); // Per task 1, OpenCV is explicitly used here
        img = edges;
    }

    // --- DATA EXTRACTION ---
    void drawHistogramTo(uchar* out_hist_data) {
        int hist[256] = {0};
        int maxCount = 0;
        int total = img.rows * img.cols;

        // Calculate
        for (int y = 0; y < img.rows; y++) {
            for (int x = 0; x < img.cols; x++) {
                int val = img.at<uchar>(y, x);
                hist[val]++;
                if (hist[val] > maxCount) maxCount = hist[val];
            }
        }

        // Draw onto 512x400 canvas (3 channels for color)
        Mat canvas(400, 512, CV_8UC3, Scalar(255, 255, 255));
        int bin_w = 2;
        Point prevPoint(0, 400);

        for (int i = 0; i < 256; i++) {
            int barHeight = cvRound(((double)hist[i] / maxCount) * 400);
            rectangle(canvas, Point(i * bin_w, 400), Point((i + 1) * bin_w, 400 - barHeight), Scalar(200, 200, 200), FILLED);

            double pdf = (double)hist[i] / total;
            int pdfHeight = cvRound((pdf * total / maxCount) * 400); 
            Point currentPoint(i * bin_w + 1, 400 - pdfHeight);
            
            if (i > 0) line(canvas, prevPoint, currentPoint, Scalar(255, 0, 0), 2, LINE_AA);
            prevPoint = currentPoint;
        }

        // Copy canvas directly to Python's memory pointer
        memcpy(out_hist_data, canvas.data, 512 * 400 * 3);
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
                      unsigned char* out_img, unsigned char* out_hist) {
        
        // 1. Initialize OOP Object
        ImageProcessor proc(in_data, rows, cols);

        // 2. Add Noise
        if (noise_type == 1) proc.addUniformNoise((int)np1, (int)np2);
        if (noise_type == 2) proc.addGaussianNoise(np1, np2);
        if (noise_type == 3) proc.addSaltAndPepper(np1, np2);

        // 3. Apply Filters & Edge Detection
        if (filter_type == 3) {
            proc.applyMedian(k_size);
        } else if (filter_type == 7) {
            proc.applyCanny();
        } else if (filter_type > 0) {
            Mat k = proc.generateKernel(filter_type, k_size, dir);
            proc.applyConvolution(k);
        }

        // 4. Output Data
        proc.copyImageTo(out_img);
        proc.drawHistogramTo(out_hist);
        
        // When this function finishes, `proc` is destroyed, preventing memory leaks!
        // When Fourier is added, you just add `proc.applyFFT()` above.
    }
}