#ifndef CARTRACKER_HPP
#define CARTRACKER_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <vector>
#include <string>

class CarTracker {
public:
    CarTracker(const std::string& cascadePath, double conversionFactor, int fps, const cv::Rect& roi);

    void initialize(cv::Mat& frame);
    void detectNewCars(cv::Mat& frame);
    void updateTrackers(cv::Mat& frame);
    int getTotalCarsDetected() const;

private:
    cv::CascadeClassifier carCascade_;
    double conversionFactor_;
    int fps_;
    int totalCarsDetected_;
    cv::Rect roi_;
    std::vector<cv::Ptr<cv::Tracker>> trackers_;
    std::vector<cv::Rect> bboxes_;
    std::vector<cv::Point2f> prevCenters_;
    std::vector<double> speeds_;

    void filterCars(const std::vector<cv::Rect>& cars, std::vector<cv::Rect>& filteredCars);
    bool isWithinFrame(const cv::Rect &bbox, const cv::Size &frameSize) const;
};

#endif // CARTRACKER_HPP
