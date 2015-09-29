/**
* @file SURF_FlannMatcher
* @brief SURF detector + descriptor + FLANN Matcher
* @author A. Huaman
*/

#include <stdio.h>
#include <iostream>
#include <cmath>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;

#define PI 3.1415926535897
#define INF 0x3f3f3f3f

#define ImgLength 4

bool cmpDMatch(const DMatch &a, const DMatch &b) {
	return a.distance < b.distance;
}

double distance(Point2f a, Point2f b) {
	return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}

double featureMatching(Mat &img_1, Mat &img_2, Mat &img_matches) {
	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;

	//SiftFeatureDetector detector(minHessian);
	SurfFeatureDetector detector(minHessian);

	std::vector<KeyPoint> keypoints_1, keypoints_2;

	detector.detect(img_1, keypoints_1);
	detector.detect(img_2, keypoints_2);

	//-- Draw keypoints
	Mat img_keypoints_1; Mat img_keypoints_2;

	drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	//-- Show detected (drawn) keypoints
	//imshow("Keypoints 1", img_keypoints_1);
	//imshow("Keypoints 2", img_keypoints_2);

	//-- Step 2: Calculate descriptors (feature vectors)
	SiftDescriptorExtractor extractor;
	//SurfDescriptorExtractor extractor;

	Mat descriptors_1, descriptors_2;

	extractor.compute(img_1, keypoints_1, descriptors_1);
	extractor.compute(img_2, keypoints_2, descriptors_2);

	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	if (sum(descriptors_2)[0] == 0) return INF;
	matcher.match(descriptors_1, descriptors_2, matches);

	sort(matches.begin(), matches.end(), cmpDMatch);
	
	double avg = 0;
	int i;
	for (i = 0; i < 20 && i < matches.size(); ++i)
		avg += distance(keypoints_1[matches[i].queryIdx].pt, keypoints_2[matches[i].trainIdx].pt);
	avg /= i;

	double diff = 0;
	for (i = 0; i < 20 && i < matches.size(); ++i)
		diff += pow(distance(keypoints_1[matches[i].queryIdx].pt, keypoints_2[matches[i].trainIdx].pt) - avg, 2);

	std::vector< DMatch > good_matches;
	for (int i = 0; i < 20 && i < matches.size(); i++){
		good_matches.push_back(matches[i]);
	}

	//-- Draw only "good" matches
	drawMatches(img_1, keypoints_1, img_2, keypoints_2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Show detected matches
	//imshow("Good Matches", img_matches);

	return diff;
}

Mat img_1, img_2;
Point p1, p2;

void getSelectRegion() {
	Point tP1, tP2;
	tP1.x = min(p1.x, p2.x); tP1.y = min(p1.y, p2.y);
	tP2.x = max(p1.x, p2.x); tP2.y = max(p1.y, p2.y);
	Mat match = img_1.colRange(tP1.x, tP2.x + 1).rowRange(tP1.y, tP2.y + 1);
	imshow("match", match);

	int w = tP2.x - tP1.x + 1;
	int h = tP2.y - tP1.y + 1;

	Mat img_matches_tmp, img_matches(h, w, match.type());
	double minV = INF;
	for (int i = 0; i < img_2.rows - h; i += h/6){
		for (int j = 0; j < img_2.cols - w; j += w/6){
			Mat input = img_2.colRange(j, j + w).rowRange(i, i + h);
			if (sum(input)[0] != 255 * w * h){
				double nowV = featureMatching(match, input, img_matches_tmp);
				if (nowV < minV){
					minV = nowV;
					img_matches = img_matches_tmp.clone();
				}
			}
		}
		std::cout << i << std::endl;
	}
	imshow("result", img_matches);
	waitKey(0);
}

bool isMouseDown = false;
void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
	if (event == EVENT_LBUTTONDOWN){
		isMouseDown = true;
		p1.x = x; p1.y = y;
	}
	else if (event == EVENT_LBUTTONUP){
		p2.x = max(0, min(x, img_1.cols - 1)); p2.y = max(0, min(y, img_1.rows - 1));
		isMouseDown = false;
		if (p1.x != p2.x && p1.y != p2.y)
			getSelectRegion();
	}
	else if (isMouseDown && event == EVENT_MOUSEMOVE){
		p2.x = max(0, min(x, img_1.cols - 1)); p2.y = max(0, min(y, img_1.rows - 1));
		Mat match_clone = img_1.clone();
		rectangle(match_clone, p1, p2, 0);
		imshow("select match", match_clone);
	}
}

Mat inputImage[ImgLength];
int main(int argc, char** argv) {

	Mat sImg_1 = imread("match.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat sImg_2 = imread("input.png", CV_LOAD_IMAGE_GRAYSCALE);

	if (!sImg_1.data || !sImg_2.data)
	{
		std::cout << " --(!) Error reading images " << std::endl; return -1;
	}

	double scale = 3;
	resize(sImg_1, img_1, Size(sImg_1.cols * scale, sImg_1.rows * scale));
	resize(sImg_2, img_2, Size(sImg_2.cols * scale, sImg_2.rows * scale));

	namedWindow("select match", WINDOW_AUTOSIZE);
	setMouseCallback("select match", CallBackFunc, NULL);
	imshow("select match", img_1);
	
	waitKey(0);

	return 0;
}