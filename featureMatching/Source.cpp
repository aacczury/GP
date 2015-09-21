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

#define PI 3.1415926535897

using namespace cv;

void readme();

void symmetryTest(
	const std::vector<std::vector<DMatch> >& matches1,
	const std::vector<std::vector<DMatch> >& matches2,
	std::vector<DMatch>& symMatches) {
	// for all matches image 1 -> image 2
	for (std::vector<std::vector<DMatch> >::
		const_iterator matchIterator1 = matches1.begin();
		matchIterator1 != matches1.end(); ++matchIterator1) {
		// ignore deleted matches
		if (matchIterator1->size() < 2)
			continue;
		// for all matches image 2 -> image 1
		for (std::vector<std::vector<DMatch> >::
			const_iterator matchIterator2 = matches2.begin();
			matchIterator2 != matches2.end();
		++matchIterator2) {
			// ignore deleted matches
			if (matchIterator2->size() < 2)
				continue;
			// Match symmetry test
			if ((*matchIterator1)[0].queryIdx ==
				(*matchIterator2)[0].trainIdx &&
				(*matchIterator2)[0].queryIdx ==
				(*matchIterator1)[0].trainIdx) {
				// add symmetrical match
				symMatches.push_back(
					DMatch((*matchIterator1)[0].queryIdx,
					(*matchIterator1)[0].trainIdx,
					(*matchIterator1)[0].distance));
				break; // next match in image 1 -> image 2
			}
		}
	}
}
int ratioTest(std::vector<std::vector<DMatch>> &matches) {
	float ratio = 0.65f;
	int removed = 0;
	// for all matches
	for (std::vector<std::vector<DMatch> >::iterator
		matchIterator = matches.begin();
		matchIterator != matches.end(); ++matchIterator) {
		// if 2 NN has been identified
		if (matchIterator->size() > 1) {
			// check distance ratio
			if ((*matchIterator)[0].distance /
				(*matchIterator)[1].distance > ratio) {
				matchIterator->clear(); // remove match
				removed++;
			}
		}
		else { // does not have 2 neighbours
			matchIterator->clear(); // remove match
			removed++;
		}
	}
	return removed;
}
cv::Mat ransacTest(
	const std::vector<cv::DMatch>& matches,
	const std::vector<cv::KeyPoint>& keypoints1,
	const std::vector<cv::KeyPoint>& keypoints2,
	std::vector<cv::DMatch>& outMatches) {
	bool refineF = true;
	double distance = 3.0;
	double confidence = 0.99;
	// Convert keypoints into Point2f
	std::vector<cv::Point2f> points1, points2;
	cv::Mat fundemental;
	for (std::vector<cv::DMatch>::
		const_iterator it = matches.begin();
		it != matches.end(); ++it) {
		// Get the position of left keypoints
		float x = keypoints1[it->queryIdx].pt.x;
		float y = keypoints1[it->queryIdx].pt.y;
		points1.push_back(cv::Point2f(x, y));
		// Get the position of right keypoints
		x = keypoints2[it->trainIdx].pt.x;
		y = keypoints2[it->trainIdx].pt.y;
		points2.push_back(cv::Point2f(x, y));
	}
	// Compute F matrix using RANSAC
	std::vector<uchar> inliers(points1.size(), 0);
	if (points1.size()>0 && points2.size()>0){
		cv::Mat fundemental = cv::findFundamentalMat(
			cv::Mat(points1), cv::Mat(points2), // matching points
			inliers,       // match status (inlier or outlier)
			CV_FM_RANSAC, // RANSAC method
			distance,      // distance to epipolar line
			confidence); // confidence probability
		// extract the surviving (inliers) matches
		std::vector<uchar>::const_iterator
			itIn = inliers.begin();
		std::vector<cv::DMatch>::const_iterator
			itM = matches.begin();
		// for all matches
		for (; itIn != inliers.end(); ++itIn, ++itM) {
			if (*itIn) { // it is a valid match
				outMatches.push_back(*itM);
			}
		}
		if (refineF) {
			// The F matrix will be recomputed with
			// all accepted matches
			// Convert keypoints into Point2f
			// for final F computation
			points1.clear();
			points2.clear();
			for (std::vector<cv::DMatch>::
				const_iterator it = outMatches.begin();
				it != outMatches.end(); ++it) {
				// Get the position of left keypoints
				float x = keypoints1[it->queryIdx].pt.x;
				float y = keypoints1[it->queryIdx].pt.y;
				points1.push_back(cv::Point2f(x, y));
				// Get the position of right keypoints
				x = keypoints2[it->trainIdx].pt.x;
				y = keypoints2[it->trainIdx].pt.y;
				points2.push_back(cv::Point2f(x, y));
			}
			// Compute 8-point F from all accepted matches
			if (points1.size()>0 && points2.size()>0){
				fundemental = cv::findFundamentalMat(
					cv::Mat(points1), cv::Mat(points2), // matches
					CV_FM_8POINT); // 8-point method
			}
		}
	}
	return fundemental;
}

bool cmpDMatch(const DMatch &a, const DMatch &b){
	return a.distance < b.distance;
}

int maint(){
	Mat img_1 = imread("match.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_2 = imread("input.png", CV_LOAD_IMAGE_GRAYSCALE);

	for (int i = 0; i < img_1.rows; ++i){
		for (int j = 0; j < img_1.cols; ++j){
			if (((uchar *)img_1.data)[i * img_1.cols + j] > 230)
				((uchar *)img_1.data)[i * img_1.cols + j] = 255;
		}
	}

	int lx, rx, ty, by;
	for (int i = 0; i < img_1.rows; ++i){
		int flag = false;
		for (int j = 0; j < img_1.cols; ++j){
			if (((uchar *)img_1.data)[i * img_1.cols + j] != 255){
				ty = i;
				flag = true;
				break;
			}
		}
		if (flag) break;
	}
	for (int j = 0; j < img_1.cols; ++j){
		int flag = false;
		for (int i = 0; i < img_1.rows; ++i){
			if (((uchar *)img_1.data)[i * img_1.cols + j] != 255){
				lx = j;
				flag = true;
				break;
			}
		}
		if (flag) break;
	}
	for (int i = img_1.rows - 1; i >= 0; --i){
		int flag = false;
		for (int j = 0; j < img_1.cols; ++j){
			if (((uchar *)img_1.data)[i * img_1.cols + j] != 255){
				by = i;
				flag = true;
				break;
			}
		}
		if (flag) break;
	}
	for (int j = img_1.cols - 1; j >= 0; --j){
		int flag = false;
		for (int i = 0; i < img_1.rows; ++i){
			if (((uchar *)img_1.data)[i * img_1.cols + j] != 255){
				rx = j;
				flag = true;
				break;
			}
		}
		if (flag) break;
	}

	int cX = 0.5*lx + 0.5*rx;
	int cY = 0.5*ty + 0.5*by;
	int maxLen = 0;
	for (int i = 0; i < img_1.rows; ++i)
		for (int j = 0; j < img_1.cols; ++j)
			if (((uchar *)img_1.data)[i * img_1.cols + j] != 255)
				if ((i - cY)*(i - cY) + (j - cX)*(j - cX) > maxLen)
					maxLen = (i - cY)*(i - cY) + (j - cX)*(j - cX);
	float cL = sqrt(sqrt(sqrt(maxLen)));
	
	Mat imS[3];
	imS[0] = img_1;
	imS[1] = img_1;
	imS[2] = img_1;
	
	Mat circleImg;
	merge(imS, 3, circleImg);
	circleImg.convertTo(circleImg, CV_32F, 1.0 / 255.0);
	circle(circleImg, Point(cX, cY), cL*cL*cL*cL, Scalar(0, 0, 255));
	circle(circleImg, Point(cX, cY), cL*cL*cL, Scalar(0, 0, 255));
	circle(circleImg, Point(cX, cY), cL*cL, Scalar(0, 0, 255));
	circle(circleImg, Point(cX, cY), cL, Scalar(0, 0, 255));

	line(circleImg, Point(lx, 0), Point(lx, circleImg.rows - 1), Scalar(255, 0, 0));
	line(circleImg, Point(rx, 0), Point(rx, circleImg.rows - 1), Scalar(255, 0, 0));
	line(circleImg, Point(0, ty), Point(circleImg.cols - 1, ty), Scalar(255, 0, 0));
	line(circleImg, Point(0, by), Point(circleImg.cols - 1, by), Scalar(255, 0, 0));

	for (int i = 0; i < 6; ++i){
		int x = cL*cL*cL*cL*cos(PI / 6 * i);
		int y = cL*cL*cL*cL*sin(PI / 6 * i);
		line(circleImg, Point(cX-x, cY-y), Point(cX+x, cY+y), Scalar(0, 200, 0));
	}

	imshow("QQ", circleImg);

	waitKey(0);
	return 0;
}

Mat img_1, img_2;
Point p1, p2;
bool isMouseDown = false;
bool isSelected = false;
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN){
		isMouseDown = true;
		p1.x = x; p1.y = y;
	}
	else if (event == EVENT_LBUTTONUP){
		p2.x = x; p2.y = y;
		isMouseDown = false;
		isSelected = true;
	}
	else if (isMouseDown && event == EVENT_MOUSEMOVE){
		p2.x = x; p2.y = y;
		Mat match_clone = img_1.clone();
		rectangle(match_clone, p1, p2, 0);
		imshow("input", match_clone);
	}
}

/**
* @function main
* @brief Main function
*/
int main(int argc, char** argv)
{
	Mat sImg_1 = imread("match.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat sImg_2 = imread("input.png", CV_LOAD_IMAGE_GRAYSCALE);

	if (!sImg_1.data || !sImg_2.data)
	{
		std::cout << " --(!) Error reading images " << std::endl; return -1;
	}

	double scale = 3;
	resize(sImg_1, img_1, Size(sImg_1.cols * scale, sImg_1.rows * scale));
	resize(sImg_2, img_2, Size(sImg_2.cols * scale, sImg_2.rows * scale));

	namedWindow("input", WINDOW_AUTOSIZE);
	setMouseCallback("input", CallBackFunc, NULL);
	imshow("input", img_2);

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;

	//SiftFeatureDetector detector(minHessian);
	SurfFeatureDetector detector(minHessian);
	//OrbFeatureDetector detector(minHessian);

	std::vector<KeyPoint> keypoints_1, keypoints_2;

	detector.detect(img_1, keypoints_1);
	detector.detect(img_2, keypoints_2);

	//-- Draw keypoints
	Mat img_keypoints_1; Mat img_keypoints_2;

	drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	//-- Show detected (drawn) keypoints
	imshow("Keypoints 1", img_keypoints_1);
	imshow("Keypoints 2", img_keypoints_2);

	//-- Step 2: Calculate descriptors (feature vectors)
	SiftDescriptorExtractor extractor;
	//SurfDescriptorExtractor extractor;
	//OrbDescriptorExtractor extractor;

	Mat descriptors_1, descriptors_2;

	extractor.compute(img_1, keypoints_1, descriptors_1);
	extractor.compute(img_2, keypoints_2, descriptors_2);

	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors_1, descriptors_2, matches);
	/*
	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< DMatch > good_matches;

	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}*/
	
	sort(matches.begin(), matches.end(), cmpDMatch);
	std::vector< DMatch > good_matches;

	for (int i = 0; i < 10 && i < matches.size(); i++)
	{
		good_matches.push_back(matches[i]);
	}
	
	//-- Draw only "good" matches
	Mat img_matches;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	/*
	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_1[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_2[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(obj, scene, CV_RANSAC);

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img_1.cols, 0);
	obj_corners[2] = cvPoint(img_1.cols, img_1.rows); obj_corners[3] = cvPoint(0, img_1.rows);
	std::vector<Point2f> scene_corners(4);

	perspectiveTransform(obj_corners, scene_corners, H);

	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line(img_matches, scene_corners[0] + Point2f(img_1.cols, 0), scene_corners[1] + Point2f(img_1.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[1] + Point2f(img_1.cols, 0), scene_corners[2] + Point2f(img_1.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[2] + Point2f(img_1.cols, 0), scene_corners[3] + Point2f(img_1.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[3] + Point2f(img_1.cols, 0), scene_corners[0] + Point2f(img_1.cols, 0), Scalar(0, 255, 0), 4);
	*/

	//-- Show detected matches
	imshow("Good Matches", img_matches);
	imwrite("output.png", img_matches);

	for (int i = 0; i < (int)good_matches.size(); i++)
	{
		printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
	}
	
	waitKey(0);

	return 0;
}

/**
* @function readme
*/
void readme()
{
	std::cout << " Usage: ./SURF_FlannMatcher <img1> <img2>" << std::endl;
}