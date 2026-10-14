#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
#include <random>
#include <map>
#include <tuple>
#include <mutex>

using namespace cv;
using namespace std;

// Export macro: required on MSVC, harmless on MinGW
#if defined(_WIN32)
  #define API __declspec(dllexport)
#else
  #define API __attribute__((visibility("default")))
#endif

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
    API void run_pipeline(unsigned char* in_data, int rows, int cols,
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

// =====================================================================
// Histogram statistics + Task 8 (color -> gray, R/G/B histograms, CDF)
// =====================================================================

// hist (256 counts) -> pdf = count / N, cdf = running sum of pdf
static void histToPdfCdf(const int* hist, float* pdf, float* cdf) {
    double total = 0.0;
    for (int i = 0; i < 256; ++i) total += hist[i];

    double running = 0.0;
    for (int i = 0; i < 256; ++i) {
        double p = (total > 0.0) ? hist[i] / total : 0.0;
        running += p;
        pdf[i] = static_cast<float>(p);
        cdf[i] = static_cast<float>(std::min(running, 1.0));
    }
}

class ColorAnalyzer {
public:
    // bgr: interleaved 8-bit, 3 channels (OpenCV order B, G, R)
    // hist / pdf / cdf: 4 x 256 values, order R, G, B, Gray
    static void run(const uchar* bgr, int rows, int cols,
                    uchar* outGray, uchar* outEqualized,
                    int* hist, float* pdf, float* cdf) {
        const size_t n = static_cast<size_t>(rows) * cols;
        fill(hist, hist + 4 * 256, 0);

        // Gray = 0.299 R + 0.587 G + 0.114 B, and the 4 histograms
        for (size_t i = 0; i < n; ++i) {
            const uchar b = bgr[3 * i];
            const uchar g = bgr[3 * i + 1];
            const uchar r = bgr[3 * i + 2];
            const uchar gray = saturate_cast<uchar>(cvRound(0.299 * r + 0.587 * g + 0.114 * b));
            outGray[i] = gray;
            hist[r]++;
            hist[256 + g]++;
            hist[512 + b]++;
            hist[768 + gray]++;
        }

        for (int c = 0; c < 4; ++c)
            histToPdfCdf(hist + c * 256, pdf + c * 256, cdf + c * 256);

        // Equalization with the gray CDF: new = round(255 * CDF[old])
        uchar lut[256];
        const int* grayHist = hist + 768;
        double cumulative = 0.0;
        for (int i = 0; i < 256; ++i) {
            cumulative += grayHist[i];
            lut[i] = saturate_cast<uchar>(cvRound(255.0 * cumulative / (double)n));
        }
        for (size_t i = 0; i < n; ++i) outEqualized[i] = lut[outGray[i]];
    }
};

extern "C" {
    API void hist_to_pdf_cdf(const int* hist, float* pdf, float* cdf) {
        histToPdfCdf(hist, pdf, cdf);
    }

    API void color_analysis(const unsigned char* bgr, int rows, int cols,
                            unsigned char* out_gray, unsigned char* out_equalized,
                            int* hist, float* pdf, float* cdf) {
        ColorAnalyzer::run(bgr, rows, cols, out_gray, out_equalized, hist, pdf, cdf);
    }
}

// =====================================================================
// Task 9 & 10: FREQUENCY DOMAIN (FFT is computed ONCE per image)
// kind: 0 = Ideal, 1 = Gaussian, 2 = Butterworth
// =====================================================================
class FreqDomain {
private:
    Mat F_;            // full complex spectrum (CV_32FC2), padded size, NOT shifted
    int rows_, cols_;  // output (original) size

    // Low-pass mask in unshifted layout (wrap-around distance). Cached.
    static Mat lowpassMask(int pr, int pc, int kind, float d0, int order) {
        static map<tuple<int, int, int, int, int>, Mat> cache;
        static mutex mtx;
        lock_guard<mutex> lock(mtx);

        if (d0 < 1.0f) d0 = 1.0f;
        auto key = make_tuple(pr, pc, kind, (int)lround(d0 * 100.0f), order);
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
        if (cache.size() > 16) cache.clear();

        Mat m(pr, pc, CV_32F);
        const float d02 = d0 * d0;
        for (int y = 0; y < pr; ++y) {
            float dy = (float)std::min(y, pr - y);
            float* row = m.ptr<float>(y);
            for (int x = 0; x < pc; ++x) {
                float dx = (float)std::min(x, pc - x);
                float d2 = dx * dx + dy * dy;
                if (kind == 0)      row[x] = (d2 <= d02) ? 1.0f : 0.0f;
                else if (kind == 1) row[x] = std::exp(-d2 / (2.0f * d02));
                else                row[x] = 1.0f / (1.0f + std::pow(std::sqrt(d2) / d0, 2.0f * order));
            }
        }
        cache.emplace(key, m);
        return m;
    }

    Mat toImage(const Mat& Fm) const {
        Mat r;
        dft(Fm, r, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
        return r(Rect(0, 0, cols_, rows_)).clone();   // crop the padding
    }

public:
    FreqDomain(const uchar* data, int rows, int cols, int outRows, int outCols)
        : rows_(outRows), cols_(outCols) {
        Mat src(rows, cols, CV_8U, const_cast<uchar*>(data));
        Mat resized;
        if (rows != outRows || cols != outCols)
            resize(src, resized, Size(outCols, outRows), 0, 0, INTER_AREA);
        else
            resized = src;

        int pr = getOptimalDFTSize(rows_), pc = getOptimalDFTSize(cols_);
        Mat f32, padded;
        resized.convertTo(f32, CV_32F);
        copyMakeBorder(f32, padded, 0, pr - rows_, 0, pc - cols_, BORDER_REPLICATE);
        dft(padded, F_, DFT_COMPLEX_OUTPUT);           // the ONLY forward FFT
    }

    // F * LP   (or F * (1 - LP) for high-pass)
    Mat filtered(int kind, float d0, int order, bool highPass) const {
        Mat lp = lowpassMask(F_.rows, F_.cols, kind, d0, order);
        Mat out(F_.size(), CV_32FC2);
        for (int y = 0; y < F_.rows; ++y) {
            const Vec2f* f = F_.ptr<Vec2f>(y);
            const float* m = lp.ptr<float>(y);
            Vec2f* o = out.ptr<Vec2f>(y);
            for (int x = 0; x < F_.cols; ++x)
                o[x] = f[x] * (highPass ? 1.0f - m[x] : m[x]);
        }
        return out;
    }

    // Hybrid: F_low * LP(dLo) + F_high * (1 - LP(dHi)); one inverse FFT later
    static Mat hybrid(const FreqDomain& lo, const FreqDomain& hi,
                      int kind, float dLo, float dHi, int order) {
        if (lo.F_.size() != hi.F_.size()) return Mat();
        Mat mLo = lowpassMask(lo.F_.rows, lo.F_.cols, kind, dLo, order);
        Mat mHi = lowpassMask(lo.F_.rows, lo.F_.cols, kind, dHi, order);
        Mat out(lo.F_.size(), CV_32FC2);
        for (int y = 0; y < out.rows; ++y) {
            const Vec2f* fl = lo.F_.ptr<Vec2f>(y);
            const Vec2f* fh = hi.F_.ptr<Vec2f>(y);
            const float* a = mLo.ptr<float>(y);
            const float* b = mHi.ptr<float>(y);
            Vec2f* o = out.ptr<Vec2f>(y);
            for (int x = 0; x < out.cols; ++x)
                o[x] = fl[x] * a[x] + fh[x] * (1.0f - b[x]);
        }
        return out;
    }

    // ---- outputs ----
    Mat lowPassU8(const Mat& Fm) const {           // clip to 0..255
        Mat u;
        toImage(Fm).convertTo(u, CV_8U);
        return u;
    }
    Mat highPassU8(const Mat& Fm) const {          // |x| stretched to 0..255
        Mat a = cv::abs(toImage(Fm)), u;
        double mx;
        minMaxLoc(a, nullptr, &mx);
        a.convertTo(u, CV_8U, mx > 0 ? 255.0 / mx : 0.0);
        return u;
    }
    Mat spectrumView(const Mat& Fm) const {        // log(1+|F|), centered, original size
        Mat p[2], mag;
        split(Fm, p);
        magnitude(p[0], p[1], mag);
        mag += 1.0f;
        cv::log(mag, mag);

        Mat shifted(mag.size(), CV_32F);           // fftshift: for display only
        int r = mag.rows, c = mag.cols;
        for (int y = 0; y < r; ++y) {
            int sy = (y + r / 2) % r;
            const float* src = mag.ptr<float>(y);
            float* dst = shifted.ptr<float>(sy);
            for (int x = 0; x < c; ++x) dst[(x + c / 2) % c] = src[x];
        }
        double mx;
        minMaxLoc(shifted, nullptr, &mx);
        Mat u8, out;
        shifted.convertTo(u8, CV_8U, mx > 0 ? 255.0 / mx : 0.0);
        resize(u8, out, Size(cols_, rows_), 0, 0, INTER_AREA);
        return out;
    }
    Mat spectrumView() const { return spectrumView(F_); }
    Mat hybridU8(const Mat& Fm) const { return lowPassU8(Fm); }
    size_t outBytes() const { return (size_t)rows_ * cols_; }
};

extern "C" {
    API void* fd_create(unsigned char* data, int rows, int cols, int out_rows, int out_cols) {
        return new FreqDomain(data, rows, cols, out_rows, out_cols);
    }
    API void fd_free(void* h) { delete static_cast<FreqDomain*>(h); }

    API void fd_spectrum(void* h, unsigned char* out) {
        auto* fd = static_cast<FreqDomain*>(h);
        Mat s = fd->spectrumView();
        memcpy(out, s.data, fd->outBytes());
    }

    // high_pass: 0 = low-pass, 1 = high-pass. out_spec may be NULL.
    API void fd_filter(void* h, int kind, float d0, int order, int high_pass,
                       unsigned char* out_img, unsigned char* out_spec) {
        auto* fd = static_cast<FreqDomain*>(h);
        Mat Fm = fd->filtered(kind, d0, order, high_pass != 0);
        Mat img = high_pass ? fd->highPassU8(Fm) : fd->lowPassU8(Fm);
        memcpy(out_img, img.data, fd->outBytes());
        if (out_spec) {
            Mat s = fd->spectrumView(Fm);
            memcpy(out_spec, s.data, fd->outBytes());
        }
    }

    // returns 1 on success, 0 if sizes mismatch
    API int fd_hybrid(void* lo, void* hi, int kind, float d_lo, float d_hi, int order,
                      unsigned char* out) {
        auto* a = static_cast<FreqDomain*>(lo);
        auto* b = static_cast<FreqDomain*>(hi);
        Mat Fm = FreqDomain::hybrid(*a, *b, kind, d_lo, d_hi, order);
        if (Fm.empty()) return 0;
        Mat img = a->hybridU8(Fm);
        memcpy(out, img.data, a->outBytes());
        return 1;
    }
}