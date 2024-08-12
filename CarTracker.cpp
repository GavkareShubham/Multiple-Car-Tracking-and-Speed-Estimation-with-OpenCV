#include "CarTracker.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <cmath>
#include <sstream>

CarTracker::CarTracker(const std::string& cascadePath, double conversionFactor, int fps, const cv::Rect& roi)
    : conversionFactor_(conversionFactor), fps_(fps), totalCarsDetected_(0), roi_(roi) {
    if (!carCascade_.load(cascadePath)) {
        std::cerr << "Error loading car cascade file" << std::endl;
        throw std::runtime_error("Failed to load Haar Cascade");
    }
}

void CarTracker::filterCars(const std::vector<cv::Rect>& cars, std::vector<cv::Rect>& filteredCars) {
    for (const auto &car : cars) {
        bool isMax = true;
        for (const auto &otherCar : cars) {
            if (&car != &otherCar && (car & otherCar).area() > 0) {
                isMax = false;
                break;
            }
        }
        if (isMax) {
            filteredCars.push_back(car);
        }
    }
}

bool CarTracker::isWithinFrame(const cv::Rect &bbox, const cv::Size &frameSize) const {
    return bbox.x >= 0 && bbox.y >= 0 &&
           bbox.x + bbox.width <= frameSize.width &&
           bbox.y + bbox.height <= frameSize.height;
}

void CarTracker::initialize(cv::Mat& frame) {
    cv::Size frameSize = frame.size();
    std::cout << "Frame size: " << frameSize.width << "x" << frameSize.height << std::endl;
    std::cout << "ROI: " << roi_ << std::endl;

    // Validate ROI dimensions
    if (roi_.x < 0 || roi_.y < 0 || roi_.width <= 0 || roi_.height <= 0 ||
        roi_.x + roi_.width > frameSize.width || roi_.y + roi_.height > frameSize.height) {
        std::cerr << "Invalid ROI dimensions" << std::endl;
        throw std::runtime_error("Invalid ROI dimensions");
    }

    cv::Mat roiFrame = frame(roi_);
    std::vector<cv::Rect> cars;
    carCascade_.detectMultiScale(roiFrame, cars, 1.1, 2, 0, cv::Size(30, 30));

    std::vector<cv::Rect> filteredCars;
    filterCars(cars, filteredCars);

    for (const auto& car : filteredCars) {
        cv::Rect adjustedCar = car + cv::Point(roi_.x, roi_.y);
        cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
        trackers_.push_back(tracker);
        bboxes_.push_back(adjustedCar);
        tracker->init(frame, adjustedCar);
        prevCenters_.push_back((adjustedCar.tl() + adjustedCar.br()) * 0.5);
        speeds_.push_back(0.0);
        totalCarsDetected_++;
    }
}

void CarTracker::detectNewCars(cv::Mat& frame) {
    cv::Size frameSize = frame.size();
    std::cout << "Frame size: " << frameSize.width << "x" << frameSize.height << std::endl;
    std::cout << "ROI: " << roi_ << std::endl;

    // Validate ROI dimensions
    if (roi_.x < 0 || roi_.y < 0 || roi_.width <= 0 || roi_.height <= 0 ||
        roi_.x + roi_.width > frameSize.width || roi_.y + roi_.height > frameSize.height) {
        std::cerr << "Invalid ROI dimensions" << std::endl;
        throw std::runtime_error("Invalid ROI dimensions");
    }

    cv::Mat roiFrame = frame(roi_);
    std::vector<cv::Rect> newCars;
    carCascade_.detectMultiScale(roiFrame, newCars, 1.1, 2, 0, cv::Size(30, 30));

    std::vector<cv::Rect> filteredNewCars;
    filterCars(newCars, filteredNewCars);

    for (const auto& car : filteredNewCars) {
        cv::Rect adjustedCar = car + cv::Point(roi_.x, roi_.y);
        bool isNew = true;
        for (const auto& bbox : bboxes_) {
            if ((adjustedCar & bbox).area() > 0) {
                isNew = false;
                break;
            }
        }

        if (isNew) {
            cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
            trackers_.push_back(tracker);
            bboxes_.push_back(adjustedCar);
            tracker->init(frame, adjustedCar);
            prevCenters_.push_back((adjustedCar.tl() + adjustedCar.br()) * 0.5);
            speeds_.push_back(0.0);
            totalCarsDetected_++;
            cv::rectangle(frame, adjustedCar, cv::Scalar(0, 255, 0), 2, 1);
        }
    }
}

void CarTracker::updateTrackers(cv::Mat& frame) {
    cv::Size frameSize = frame.size();
    std::cout << "Frame size: " << frameSize.width << "x" << frameSize.height << std::endl;

    for (size_t i = 0; i < trackers_.size();) {
        bool ok = trackers_[i]->update(frame, bboxes_[i]);

        if (ok && isWithinFrame(bboxes_[i], frameSize)) {
            cv::rectangle(frame, bboxes_[i], cv::Scalar(255, 0, 0), 2, 1);
            cv::Point2f currentCenter = (bboxes_[i].tl() + bboxes_[i].br()) * 0.5;
            double distance = cv::norm(currentCenter - prevCenters_[i]);
            double distance_meters = distance * conversionFactor_;
            double speed_mps = distance_meters * fps_;
            double speed_kmh = round(speed_mps * 3.6);
            speeds_[i] = speed_kmh;

            cv::putText(frame, "Speed: " + std::to_string((int)speed_kmh) + " km/h",
                        bboxes_[i].tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 255, 0), 2);

            prevCenters_[i] = currentCenter;
            ++i;
        } else {
            trackers_.erase(trackers_.begin() + i);
            bboxes_.erase(bboxes_.begin() + i);
            prevCenters_.erase(prevCenters_.begin() + i);
            speeds_.erase(speeds_.begin() + i);
        }
    }
}

int CarTracker::getTotalCarsDetected() const {
    return totalCarsDetected_;
}
