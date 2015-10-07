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

#define ImgLength 5
int imgIndex = 0;

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

Point selectRegion[ImgLength][5][2];
void match(){

}

Point selectRegionCenter[ImgLength];
double Lw[ImgLength][ImgLength];

double distanceSquare(Point a, Point b) {
	return (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y);
}
/**	Lw = Dw - Aw 
*	Dw = sum(w(i,.))
*	Aw = w(i,j)
*/
void ringIt(){
	// - Aw
	double t = 10322;
	for (int i = 0; i < ImgLength; ++i){
		for (int j = i; j < ImgLength; ++j){
			Lw[i][j] = Lw[j][i] = -exp(-distanceSquare(selectRegionCenter[i], selectRegionCenter[j]) / t);
			std::cout << distanceSquare(selectRegionCenter[i], selectRegionCenter[j]) << ", ";
		}
		std::cout << std::endl;
	}

	// Lw = - Aw - (-Dw)
	for (int i = 0; i < ImgLength; ++i){
		double sumD = 0;
		// -Dw
		for (int j = 0; j < ImgLength; ++j){
			sumD += Lw[i][j];
		}
		Lw[i][i] -= sumD;
	}

	for (int i = 0; i < ImgLength; ++i){
		for (int j = 0; j < ImgLength; ++j)
			std::cout << Lw[i][j] << ", ";
		std::cout << std::endl;
	}
	Mat e = Mat(ImgLength, ImgLength, CV_32FC1);
	for (int i = 0; i < ImgLength; ++i)
		for (int j = 0; j < ImgLength; ++j)
			((float *)e.data)[i * e.cols + j] = (float)Lw[i][j];
	std::cout << e << std::endl;
	Mat eVal, eVec;
	eigen(e, eVal, eVec);
	std::cout << eVec << std::endl;

	Mat ring = Mat(500, 500, CV_32FC1, Scalar(1));
	int c = 250, r = 200;
	circle(ring, Point(250, 250), 1, 0, 2);
	for (int i = 0; i < ImgLength; ++i){
		circle(ring, Point(250 + r*((float *)eVec.data)[(ImgLength - 2) * e.cols + i], 250 + r*((float *)eVec.data)[(ImgLength - 3) * e.cols + i]), 1, 0, 2);
		char text[10];
		itoa(i, text, 10);
		putText(ring, string(text), Point(250 + r*((float *)eVec.data)[(ImgLength - 2) * e.cols + i], 250 + r*((float *)eVec.data)[(ImgLength - 3) * e.cols + i]), FONT_HERSHEY_DUPLEX, 1, 0);
	}
	imshow("RingIt", ring);
}

Mat inputImage[ImgLength];
Point p1, p2;
int selectCount = 0;
void getSelectRegion() {
	Point tP1, tP2;
	tP1.x = min(p1.x, p2.x); tP1.y = min(p1.y, p2.y);
	tP2.x = max(p1.x, p2.x); tP2.y = max(p1.y, p2.y);
	selectRegion[imgIndex][selectCount][0] = tP1;
	selectRegion[imgIndex][selectCount++][1] = tP2;

	if (selectCount == 5){
		selectCount = 0;
		imgIndex++;
		if (imgIndex < ImgLength)
			imshow("select region", inputImage[imgIndex]);
		else
			match();
	}
	//selectRegionCenter[imgIndex] = Point((tP1.x + tP2.x) / 2, (tP1.y + tP2.y) / 2);
	//printf("(%d, %d)\n", selectRegionCenter[imgIndex].x, selectRegionCenter[imgIndex].y);
	//imgIndex++;
	//if (imgIndex < ImgLength)
	//	imshow("select region", inputImage[imgIndex]);
	//else
	//	ringIt();
}

bool isMouseDown = false;
void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
	if (event == EVENT_LBUTTONDOWN){
		isMouseDown = true;
		p1.x = x; p1.y = y;
	}
	else if (event == EVENT_LBUTTONUP){
		p2.x = max(0, min(x, inputImage[imgIndex].cols - 1)); p2.y = max(0, min(y, inputImage[imgIndex].rows - 1));
		isMouseDown = false;
		if (p1.x != p2.x && p1.y != p2.y)
			getSelectRegion();
	}
	else if (isMouseDown && event == EVENT_MOUSEMOVE){
		p2.x = max(0, min(x, inputImage[imgIndex].cols - 1)); p2.y = max(0, min(y, inputImage[imgIndex].rows - 1));
		Mat selectRegion = inputImage[imgIndex].clone();
		rectangle(selectRegion, p1, p2, 0);
		imshow("select region", selectRegion);
	}
}

int main(int argc, char** argv) {
	double scale = 3;
	Mat readImg;
	for (int i = 0; i < ImgLength; ++i){
		char filename[10];
		sprintf(filename, "Eren/%02d.png", i + 1);

		readImg = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
		resize(readImg, inputImage[i], Size(readImg.cols * scale, readImg.rows * scale));
	}

	imgIndex = 0;
	namedWindow("select region", WINDOW_AUTOSIZE);
	setMouseCallback("select region", CallBackFunc, NULL);
	imshow("select region", inputImage[imgIndex]);
	
	waitKey(0);

	return 0;
}