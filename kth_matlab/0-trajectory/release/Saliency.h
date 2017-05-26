//--------------------------------begin--------------------------------
//-----------------------------by Yikun Lin----------------------------
#ifndef SALIENCY_H_
#define SALIENCY_H_

#include <stdio.h>
#include "DenseTrackStab.h"
class Saliency
{
public:
	Saliency();
	virtual ~Saliency();
	float CalcStaticSaliencyMap(
		Mat& grey_mat, // grey image
		CvMat* salMap, // OUTPUT: saliency map
		const bool& normflag = true); // false if normalization is not needed
	float CalcMotionSaliencyMap(
		Mat& flow_mat,  // optical flow image
		const DescInfo& descInfo, // parameters about the descriptor
		CvMat* kernelMatrix, // kernel matrix of densely sampled angles with bins
		CvMat* salMap, // OUTPUT: saliency map
		const bool& normflag = true); // false if normalization is not needed
	
private:
	void GaussianStaticSmooth(
		IplImage* inputImg,
		const std::vector<float>& kernel,
		IplImage* smoothImg);
	void GaussianMotionSmooth(
		IplImage* inputImg,
		const std::vector<float>& kernel,
		IplImage* smoothImg);
	void CreateStaticIntegralImage(
		IplImage* inputImg,
		float* intImg);
	void CreateMotionIntegralImage(
		IplImage* inputImg,
		const DescInfo& descInfo,
		CvMat* kernelMatrix,
		float* intImg);
	float GetStaticIntegralSum(
		const float* intImg,
		int x1,
		int y1,
		int x2,
		int y2,
		const int& height,
		const int& width);
	float GetMotionIntegralSum(
		const float* intImg,
		int x1,
		int y1,
		int x2,
		int y2,
		const int& height,
		const int& width,
		const int& nBins,
		std::vector<float>& hist);
	float Normalize(
		CvMat* salMap,
		const int& normrange = 255);
};
#endif /*SALIENCY_H_*/
//---------------------------------end---------------------------------
