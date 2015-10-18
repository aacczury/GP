/**
* @file SURF_FlannMatcher
* @brief SURF detector + descriptor + FLANN Matcher
* @author A. Huaman
*/

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <ctime>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;

#define PI 3.1415926535897
#define INF 0x3f3f3f3f
#define INPUTPATH "test/%02d.png"
#define ImgLength 2
#define SCALE 3

#define FindArea 200

#define FeatureNum 4
int imgIndex = 0;
Mat inputImage[ImgLength];

bool cmpDMatch(const DMatch &a, const DMatch &b) {
	return a.distance < b.distance;
}

double distance(Point2f a, Point2f b) {
	return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}

Point2f avgPoint2f(std::vector<Point2f> a){
	Point2f avgA(0, 0);
	for (int i = 0; i < a.size(); ++i)
		avgA += a[i];
	avgA.x /= a.size();
	avgA.y /= a.size();

	return avgA;
}

double corr2(std::vector<Point2f> a, std::vector<Point2f> b){
	Point2f avgA = avgPoint2f(a), avgB = avgPoint2f(b);
	//std::cout << avgA.x << ", " << avgA.y << std::endl;
	double varA = 0, varB = 0, varAB = 0;
	for (int i = 0; i < a.size(); ++i){
		varA += (a[i] - avgA).ddot(a[i] - avgA);
		varB += (b[i] - avgB).ddot(b[i] - avgB);
		varAB += (a[i] - avgA).ddot(b[i] - avgB);
	}
	//std::cout << varAB << "\t" << varA << "\t" << varB << std::endl;
	if (varA == 0 || varB == 0)
		return -2;
	return varAB / sqrt(varA * varB);
}

double featureMatching(Mat &img_1, Mat &img_2, Mat &img_matches, Point2f &center) {
	double diff = -2;
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
	if (sum(descriptors_2)[0] == 0) return diff;
	matcher.match(descriptors_1, descriptors_2, matches);

	sort(matches.begin(), matches.end(), cmpDMatch);
	
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int i = 0; i < 20 && i < matches.size(); ++i){
		obj.push_back(keypoints_1[matches[i].queryIdx].pt);
		scene.push_back(keypoints_2[matches[i].trainIdx].pt);
	}

	diff = corr2(obj, scene);
	center = avgPoint2f(scene);
	
	//std::cout << diff << std::endl;
	
	//Mat H = findHomography(obj, scene, 0);

	/*std::vector<Point2f> transObj(obj.size());
	perspectiveTransform(obj, transObj, H);
	double diff;
	for (int i = 0; i < 20 && i < matches.size(); ++i)
		diff += distance(transObj[i], scene[i]);
	*/

	/*
	double avg = 0;
	int i;
	for (i = 0; i < 20 && i < matches.size(); ++i)
		avg += distance(keypoints_1[matches[i].queryIdx].pt, keypoints_2[matches[i].trainIdx].pt);
	avg /= i;

	double diff = 0;
	for (i = 0; i < 20 && i < matches.size(); ++i)
		diff += pow(distance(keypoints_1[matches[i].queryIdx].pt, keypoints_2[matches[i].trainIdx].pt) - avg, 2);
	*/

	//-- Draw only "good" matches
	//std::vector< DMatch > good_matches;
	//for (int i = 0; i < 20 && i < matches.size(); i++){
	//	good_matches.push_back(matches[i]);
	//}
	//drawMatches(img_1, keypoints_1, img_2, keypoints_2,
	//	good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
	//	vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Show detected matches
	//imshow("Good Matches", img_matches);

	return diff;
}

Point selectRegion[ImgLength][FeatureNum][2];
int matchOrder[ImgLength][FeatureNum];

void showMatchResult(){
	Mat splitShowImg[ImgLength][3];
	Mat showImg[ImgLength];
	for (int i = 0; i < ImgLength; ++i){
		splitShowImg[i][0] = splitShowImg[i][1] = splitShowImg[i][2] = inputImage[i];
		merge(splitShowImg[i], 3, showImg[i]);
	}

	srand(time(NULL));
	for (int i = 0; i < FeatureNum; ++i){
		Scalar color(rand() % 200, rand() % 200, rand() % 200);
		for (int j = 0; j < ImgLength; ++j){
			Point p1 = selectRegion[j][matchOrder[j][i]][0];
			Point p2 = selectRegion[j][matchOrder[j][i]][1];
			rectangle(showImg[j], Rect(p1, p2), color, 2);
		}
	}
	for (int i = 0; i < ImgLength; ++i){
		char titleName[10];
		sprintf(titleName, "match img %d", i + 1);
		imshow(titleName, showImg[i]);
		sprintf(titleName, "match img %d.png", i + 1);
		imwrite(titleName, showImg[i]);
	}
}

void minArray(double *a, int length, double &minV, int &minI){
	minV = INF;
	minI = 0;
	for (int i = 0; i < length; ++i){
		if (minV > a[i]){
			minV = a[i];
			minI = i;
		}
	}
	return;
}

bool isDump(int *a, int length, int n){
	for (int i = 0; i < length; ++i)
		if (a[i] == n)
			return true;
	return false;
}

int findEmpty(double *a, int *b, int length, int index){
	double minV = INF;
	int minI = index;
	b[index] = -1;
	do{
		a[minI] = INF;
		minArray(a, length, minV, minI);
	} while (isDump(b, length, minI));
	return minI;
}

void match(){
	Mat imgSelect[ImgLength][FeatureNum];
	for (int i = 0; i < ImgLength; ++i){
		for (int j = 0; j < FeatureNum; ++j){
			Point p1 = selectRegion[i][j][0];
			Point p2 = selectRegion[i][j][1];
			imgSelect[i][j] = inputImage[i].colRange(p1.x, p2.x + 1).rowRange(p1.y, p2.y + 1);
		}
	}

	for (int i = 0; i < FeatureNum; ++i)
		matchOrder[0][i] = i;

	Point2f center;
	for (int i = 0; i < ImgLength - 1; ++i){
		double diff[FeatureNum][FeatureNum];
		Mat bast_matches[FeatureNum][FeatureNum];
		for (int j = 0; j < FeatureNum; ++j){
			for (int k = 0; k < FeatureNum; ++k){
				Mat img_matches;
				diff[j][k] = diff[k][j] = featureMatching(imgSelect[i][j], imgSelect[i + 1][k], img_matches, center);
				bast_matches[j][k] = bast_matches[k][j] = img_matches.clone();
			}
		}
		for (int j = 0; j < FeatureNum; ++j){
			double minV = INF;
			int mink = 0;
			for (int k = 0; k < FeatureNum; ++k){
				if (minV > diff[j][k]){
					minV = diff[j][k];
					mink = k;
				}
			}
			matchOrder[i + 1][j] = mink;
		}

		int dump[FeatureNum][2] = { 0 };
		for (int j = 0; j < FeatureNum; ++j){
			if (!dump[matchOrder[i + 1][j]][0]){
				dump[matchOrder[i + 1][j]][0] = 1;
				dump[matchOrder[i + 1][j]][1] = j;
			}
			else{
				int d1 = dump[matchOrder[i + 1][j]][1];
				int d2 = j;
				double tmp1[FeatureNum], tmp2[FeatureNum];
				for (int k = 0; k < FeatureNum; ++k){
					tmp1[k] = diff[j][d1];
					tmp2[k] = diff[j][d2];
				}

				double minV1, minV2;
				int minI1, minI2;
				minArray(tmp1, FeatureNum, minV1, minI1);
				minArray(tmp2, FeatureNum, minV2, minI2);

				int order[FeatureNum];
				for (int k = 0; k < FeatureNum; ++k)
					order[k] = matchOrder[i + 1][k];
				if (minV1 < minV2)
					matchOrder[i + 1][d2] = findEmpty(tmp2, order, FeatureNum, d2);
				else
					matchOrder[i + 1][d1] = findEmpty(tmp1, order, FeatureNum, d1);

				dump[matchOrder[i + 1][d1]][0] = 1;
				dump[matchOrder[i + 1][d1]][1] = d1;
				dump[matchOrder[i + 1][d2]][0] = 1;
				dump[matchOrder[i + 1][d2]][1] = d2;
			}
		}
	}
	showMatchResult();
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
	double t = 100000;
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

	Mat ring = Mat(1000, 1000, CV_32FC1, Scalar(1));
	int c = 500, r = 500;
	circle(ring, Point(c, c), 1, 0, 2);
	for (int i = 0; i < ImgLength; ++i){
		circle(ring, Point(c + r*((float *)eVec.data)[(ImgLength - 2) * e.cols + i], c + r*((float *)eVec.data)[(ImgLength - 3) * e.cols + i]), 1, 0, 2);
		char text[10];
		itoa(i, text, 10);
		putText(ring, string(text), Point(c + r*((float *)eVec.data)[(ImgLength - 2) * e.cols + i], c + r*((float *)eVec.data)[(ImgLength - 3) * e.cols + i]), FONT_HERSHEY_DUPLEX, 1, 0);
	}
	imshow("RingIt", ring);
}

Point findArea(Point tP1, Point tP2, Mat &img1, Mat &img2){
	Mat match = img1
		.colRange(max(0, tP1.x), min(img1.cols, tP2.x + 1))
		.rowRange(max(0, tP1.y), min(img1.rows, tP2.y + 1));
	imshow("match", match);
	int maxL = max(tP2.x - tP1.x, tP2.y - tP1.y);
	int w = maxL + 1;
	int h = maxL + 1;

	Point searchP1(max(0, tP1.x - w), max(0, tP1.y - h));
	Point searchP2(min(img2.cols, tP2.x + w), min(img2.rows, tP2.y + h));

	Mat img_matches_tmp/*, img_matches*/;
	Point2f match_center;
	int maxCount = 1;
	double maxV = -2;
	Point maxP(0, 0);
	for (int i = searchP1.y; i < searchP2.y - h; i += h / h){
		for (int j = searchP1.x; j < searchP2.x - w; j += w / w){
			Mat input = img2.colRange(j, j + w).rowRange(i, i + h);
			if (sum(input)[0] != 255 * w * h){
				double nowV = featureMatching(match, input, img_matches_tmp, match_center);
				if (nowV > maxV){
					maxCount = 1;
					maxV = nowV;
					maxP.x = match_center.x + j;
					maxP.y = match_center.y + i;
					//img_matches = img_matches_tmp.clone();
				}
				else if (nowV == maxV){
					maxCount ++;
					maxP.x += match_center.x + j;
					maxP.y += match_center.y + i;
				}
			}
		}

		std::cout << i << ", " << maxV << std::endl;
		std::cout << maxP.x / maxCount << ", " << maxP.y / maxCount << std::endl;
	}
	maxP.x /= maxCount;
	maxP.y /= maxCount;
	Mat img_matches = img2
		.colRange(max(0, maxP.x - w / 2 + 1), min(img2.cols, maxP.x + w / 2))
		.rowRange(max(0, maxP.y - h / 2 + 1), min(img2.rows, maxP.y + h / 2));
	imshow("result", img_matches);
	waitKey(0);

	return maxP;
}

/*
a <> b
c <> a <> b
c <> a <> b <> d
*/
typedef struct ImgNode{
	int left, right;
	Point center;
}imgNode;

typedef struct SortNode{
	Point center;
	int index;
	double value;
}sortNode;

bool sortNodeCompare(sortNode a, sortNode b){
	return a.value < b.value;
}

void findMatchCircle(Point tP1, Point tP2){
	imgNode imgRing[ImgLength + 1];
	for (int i = 0; i < ImgLength + 1; ++i){
		imgRing[i].left = ImgLength;
		imgRing[i].right = ImgLength;
		imgRing[i].center = Point(0, 0);
	}

	int maxL = max(tP2.x - tP1.x, tP2.y - tP1.y);
	int w = maxL + 1;
	int h = maxL + 1;
	Point diffArea(w / 2, h / 2);

	int curr = 0;
	imgRing[0].center = Point((tP1.x + tP2.x) / 2, (tP1.y + tP2.y) / 2);
	while (1){
		/*¦V¥k*/
		vector<sortNode> diff;
		for (int i = 0; i < ImgLength; ++i){
			if (i == curr || i == imgRing[curr].left) continue;
			sortNode tmpSortNode;
			tmpSortNode.index = i;
			tmpSortNode.center = findArea(imgRing[curr].center - diffArea, imgRing[curr].center + diffArea, inputImage[curr], inputImage[i]);
			tmpSortNode.value = distanceSquare(tmpSortNode.center, imgRing[curr].center);
			diff.push_back(tmpSortNode);
		}
		sort(diff.begin(), diff.end(), sortNodeCompare);
		if (diff[0].index == imgRing[imgRing[curr].left].left)
			break;
		else{
			imgRing[curr].right = diff[0].index;
			imgRing[imgRing[curr].right].left = curr;
			curr = imgRing[curr].right;
			imgRing[curr].center = diff[0].center;
		}
		for (int j = 0; j != ImgLength; j = imgRing[j].right)
			std::cout << j << ", ";
		std::cout << std::endl;
	}
}

Point p1, p2;
int selectCount = 0;
void getSelectRegion() {
	Point tP1, tP2;
	tP1.x = min(p1.x, p2.x); tP1.y = min(p1.y, p2.y);
	tP2.x = max(p1.x, p2.x); tP2.y = max(p1.y, p2.y);
	//selectRegion[imgIndex][selectCount][0] = tP1;
	//selectRegion[imgIndex][selectCount++][1] = tP2;

	//if (selectCount == FeatureNum){
	//	selectCount = 0;
	//	imgIndex++;
	//	if (imgIndex < ImgLength)
	//		imshow("select region", inputImage[imgIndex]);
	//	else
	//		match();
	//}

	//selectRegionCenter[imgIndex] = Point((tP1.x + tP2.x) / 2, (tP1.y + tP2.y) / 2);
	//printf("(%d, %d)\n", selectRegionCenter[imgIndex].x, selectRegionCenter[imgIndex].y);
	//imgIndex++;
	//if (imgIndex < ImgLength)
	//	imshow("select region", inputImage[imgIndex]);
	//else
	//	ringIt();
	findMatchCircle(tP1, tP2);
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

void calShapeContext(Point center, double R, Mat &shapeBins){
	
	shapeBins = Mat::zeros(6, 12, CV_16U); // 6th bins is out of range 6 * 12 * unsigned short
	Point begin(max(0, int(center.x - pow(R, 5) - 1)), max(0, int(center.y - pow(R, 5) - 1)));
	Point end(min(inputImage[imgIndex].cols - 1, int(center.x + pow(R, 5) + 1)), min(inputImage[imgIndex].rows - 1, int(center.y + pow(R, 5) + 1)));
	
	for (int i = begin.y; i < end.y; ++i){
		for (int j = begin.x; j < end.x; ++j){
			double r = sqrt(distanceSquare(Point(j, i), center));
			if (r == 0) continue;
			int d = 0;
			if (r > pow(R, 1)) d++;
			if (r > pow(R, 2)) d++;
			if (r > pow(R, 3)) d++;
			if (r > pow(R, 4)) d++;
			if (r > pow(R, 5)) d++;
			double a = acos(abs(j - center.x) / r);
			if (j < center.x) a += PI / 2;
			if (i > center.y) a = 2 * PI - a;
			if (int(a * 6 / PI) > 11)
				printf("%d\n", int(a * 6 / PI));
			if (((uchar *)inputImage[imgIndex].data)[i * inputImage[imgIndex].cols + j] < 200)
				((ushort *)shapeBins.data)[d * shapeBins.cols + int(a * 6 / PI)] ++;
		}
	}
}

void showShapeRange(Point center, double R, Mat img, string windowsName){
	Mat selectRegionSplit[3];
	selectRegionSplit[0] = img.clone();
	selectRegionSplit[1] = img.clone();
	selectRegionSplit[2] = img.clone();
	
	Mat selectRegion;
	merge(selectRegionSplit, 3, selectRegion);
	circle(selectRegion, center, pow(R, 1), Scalar(0, 0, 255));
	circle(selectRegion, center, pow(R, 2), Scalar(0, 0, 255));
	circle(selectRegion, center, pow(R, 3), Scalar(0, 0, 255));
	circle(selectRegion, center, pow(R, 4), Scalar(0, 0, 255));
	circle(selectRegion, center, pow(R, 5), Scalar(0, 0, 255));

	imshow(windowsName, selectRegion);
}


void showRectRange(Point P1, Point P2, Mat img, string windowsName){
	Mat selectRegionSplit[3];
	selectRegionSplit[0] = img.clone();
	selectRegionSplit[1] = img.clone();
	selectRegionSplit[2] = img.clone();

	Mat selectRegion;
	merge(selectRegionSplit, 3, selectRegion);
	rectangle(selectRegion, P1, P2, Scalar(0, 0, 255));

	imshow(windowsName, selectRegion);
}

double getShapeDiff(Mat s1, Mat s2){
	double diff = 0;
	for (int i = 0; i < 5; ++i){
		for (int j = 0; j < 12; ++j){
			if (((ushort *)s1.data)[i * s1.cols + j] + ((ushort *)s2.data)[i * s2.cols + j] != 0){
				diff += ((double)((ushort *)s1.data)[i * s1.cols + j] - (double)((ushort *)s2.data)[i * s2.cols + j])
					* ((double)((ushort *)s1.data)[i * s1.cols + j] - (double)((ushort *)s2.data)[i * s2.cols + j])
					/ ((double)((ushort *)s1.data)[i * s1.cols + j] + (double)((ushort *)s2.data)[i * s2.cols + j]);
			}
		}
	}

	return diff / 2;
}

void findSimilarShape(){
	double R = 2.4; // need robust
	double range = 2.5;
	Point center = Point(p2.x, p2.y);

	Mat initImgShape;
	calShapeContext(center, R, initImgShape);
	showShapeRange(center, R, inputImage[imgIndex], "select region");

	imgIndex++;
	Point begin(max(int(pow(R, 5)), int(center.x - range * pow(R, 5) - 1)), max(int(pow(R, 5)), int(center.y - range * pow(R, 5) - 1)));
	Point end(min(int(inputImage[imgIndex].cols - pow(R, 5)), int(center.x + range * pow(R, 5) + 1)), min(int(inputImage[imgIndex].rows - pow(R, 5)), int(center.y + range * pow(R, 5) + 1)));
	//showRectRange(begin, end, inputImage[imgIndex], "QQQ");
	
	double minV = INF;
	Point minCenter(0, 0);
	Mat perShape;
	for (int i = begin.y; i < end.y; ++i){
		for (int j = begin.x; j < end.x; ++j){
			calShapeContext(Point(j, i), R, perShape);

			double diff = getShapeDiff(initImgShape, perShape);
			if (diff <= minV){
				minV = diff;
				minCenter = Point(j, i);
			}
		}
		printf("%4d: %lf\n", i, minV);
	}
	showShapeRange(minCenter, R, inputImage[imgIndex], "QQQQQQQQQQQ");
}

void CallBackFunct(int event, int x, int y, int flags, void* userdata) {
	if (event == EVENT_LBUTTONUP){
		p2.x = max(0, min(x, inputImage[imgIndex].cols - 1)); p2.y = max(0, min(y, inputImage[imgIndex].rows - 1));
		findSimilarShape();
	}
}


int main(int argc, char** argv) {
	Mat readImg;
	for (int i = 0; i < ImgLength; ++i){
		char filename[10];
		sprintf(filename, INPUTPATH, i + 1);

		readImg = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
		resize(readImg, inputImage[i], Size(readImg.cols * SCALE, readImg.rows * SCALE));
	}
	
	imgIndex = 0;
	namedWindow("select region", WINDOW_AUTOSIZE);
	setMouseCallback("select region", CallBackFunct, NULL);
	imshow("select region", inputImage[imgIndex]);
	
	waitKey(0);

	return 0;
}