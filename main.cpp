#include <opencv2/opencv.hpp>
#include <iostream>
#include "CarTracker.hpp"

int main() {
    try {
        // Parameters
        std::string cascadePath = "haarcascade_car.xml";
        double conversionFactor = 0.05; // Example conversion factor (pixels to meters)
        int fps = 40; // Assume video has a frame rate of 40 FPS

        // Open video file
        cv::VideoCapture video("Cars.mp4");
        if (!video.isOpened()) {
            std::cerr << "Could not read video file" << std::endl;
            return 1;
        }

        cv::Mat frame;
        if (!video.read(frame)) {
            std::cerr << "Failed to read the first frame" << std::endl;
            return 1;
        }

        // Get frame size and adjust ROI accordingly
        cv::Size frameSize = frame.size();
        cv::Rect roi(100, 100, frameSize.width - 200, frameSize.height - 200); // Adjust ROI size as needed

        // Create CarTracker instance
        CarTracker carTracker(cascadePath, conversionFactor, fps, roi);

        // Initialize tracker
        carTracker.initialize(frame);

        while (video.read(frame)) {
            carTracker.updateTrackers(frame);
            carTracker.detectNewCars(frame);

            cv::rectangle(frame, roi, cv::Scalar(0, 255, 255), 2, 1);
            std::stringstream ss;
            ss << "Current Cars: " << carTracker.getTotalCarsDetected();
            cv::putText(frame, ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

            double accuracy = (carTracker.getTotalCarsDetected() / 40.0) * 100;
            std::stringstream ssTotal;
            ssTotal << "Total Cars: " << carTracker.getTotalCarsDetected() << " (Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%)";
            cv::putText(frame, ssTotal.str(), cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);

            cv::imshow("Tracking", frame);

            if (cv::waitKey(1) == 27) { // Exit if ESC pressed
                break;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
