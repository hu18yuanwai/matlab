#include "Saliency.h"
//--------------------------------begin--------------------------------
//-----------------------------by Yikun Lin----------------------------
//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

Saliency::Saliency()
{

}

Saliency::~Saliency()
{

}

//==============================================================================
///	Normalize
//==============================================================================
float Saliency::Normalize(
	CvMat* salMap,
	const int& normrange)
{
	int width = salMap->width;
	int height = salMap->height;
	
	
	float maxVal(0);
	float minVal(FLT_MAX);
	
	for(int iHeight = 0; iHeight < height; iHeight++) {
		float* pMap = (float*)(salMap->data.ptr + iHeight * salMap->step);
		for(int iWidth = 0; iWidth < width; iWidth++) {
			if (maxVal < pMap[iWidth])
			{
				maxVal = pMap[iWidth];
			}
			if (minVal > pMap[iWidth])
			{
				minVal = pMap[iWidth];
			}
		}

	}
	
	float range = maxVal - minVal;

	if (0 == range)
	{
		range = 1;
	} 
	float averageSaliency = 0;
	for(int iHeight = 0; iHeight < height; iHeight++) {
		float* pMap = (float*)(salMap->data.ptr + iHeight * salMap->step);
		for(int iWidth = 0; iWidth < width; iWidth++) {
			pMap[iWidth] = normrange * (pMap[iWidth] - minVal) / range;
			averageSaliency += pMap[iWidth];
		}
	}
	
	return averageSaliency;
}


//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// static saliency map.
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//==============================================================================
///	GaussianStaticSmooth
///
///	Blur a grey image with a separable binomial kernel passed in.
//==============================================================================
void Saliency::GaussianStaticSmooth(
		IplImage* inputImg,
		const std::vector<float>& kernel,
		IplImage* smoothImg)
{
	int width = inputImg->width;
	int height = inputImg->height;
	int center = kernel.size() / 2;
	IplImage* tempImg = cvCreateImage(cvGetSize(inputImg), IPL_DEPTH_32F, 1);
	
	//--------------------------------------------------------------------------
   	// Blur in the x direction.
   	//---------------------------------------------------------------------------
	for (int iHeight = 0; iHeight < height; iHeight++) {
		for (int iWidth = 0; iWidth < width; iWidth++) {
			float kernelSum(0);
			float sum(0);
			const uchar* pInput = (const uchar*)(inputImg->imageData + iHeight * inputImg->widthStep);
			float* pTemp = (float*)(tempImg->imageData + iHeight * tempImg->widthStep);
			for (int ww = (-center); ww <= center; ww++) {
				if(((iWidth + ww) >= 0) && ((iWidth + ww) < width))
				{
					sum += ((float)pInput[iWidth + ww]) * kernel[center + ww];
					kernelSum += kernel[center + ww];
				}
			}
			pTemp[iWidth] = sum / kernelSum;
		}
	}
	
	//--------------------------------------------------------------------------
   	// Blur in the y direction.
   	//---------------------------------------------------------------------------
	for (int iHeight = 0; iHeight < height; iHeight++) {
		for (int iWidth = 0; iWidth < width; iWidth++) {
			float kernelSum(0);
			float sum(0);
			float* pSmooth = (float*)(smoothImg->imageData + iHeight * smoothImg->widthStep);
			for (int hh = (-center); hh <= center; hh++) {
				if(((iHeight + hh) >= 0) && ((iHeight + hh) < height))
				{
					float* pTemp = (float*)(tempImg->imageData + (iHeight + hh) * tempImg->widthStep);
					sum += pTemp[iWidth] * kernel[center + hh];
					kernelSum += kernel[center + hh];
				}
			}
			pSmooth[iWidth] = sum / kernelSum;
		}
	}
	cvReleaseImage(&tempImg);
}

//==============================================================================
///	CreateStaticIntegralImage
//==============================================================================
void Saliency::CreateStaticIntegralImage(
	IplImage* inputImg,
	float* intImg)
{
	int width = inputImg->width;
	int height = inputImg->height;
	int index = 0;

	for(int iHeight = 0; iHeight < height; iHeight++) {
		const float* f = (const float*)(inputImg->imageData + inputImg->widthStep * iHeight);
		// the greay value accumulated in the current line
		float sum(0);
		for(int iWidth = 0; iWidth < width; iWidth++, index++) {
			sum += f[iWidth];			
			
			int temp0 = index;
			if(0 == iHeight) { // for the first line
				intImg[temp0] = sum;
			}
			else {
				int temp1 = index - width;
				intImg[temp0] = intImg[temp1] + sum;
			}
		}
	}
}

//==============================================================================
///	GetStaticIntegralSum
//==============================================================================
float Saliency::GetStaticIntegralSum(
	const float* intImg,
	int x1,
	int y1,
	int x2,
	int y2,
	const int& height,
	const int& width)
{
	x1 = std::max<int>(x1, 0);
	y1 = std::max<int>(y1, 0);
	x2 = std::min<int>(x2, width - 1);
	y2 = std::min<int>(y2, height - 1);
	int TopLeft = (y1 - 1) * width + (x1 - 1);
	int TopRight = (y1 - 1) * width + x2;
	int BottomLeft = y2 * width + (x1 - 1);
	int BottomRight = y2 * width + x2; 
	
	
	float sumTopLeft(0), sumTopRight(0), sumBottomLeft(0), sumBottomRight(0);
	if (y1 - 1 >= 0) 
	{
		if (x1 - 1 >= 0)
			sumTopLeft = intImg[TopLeft];
		if (x2 >= 0)
			sumTopRight = intImg[TopRight];
	}
	if (y2 >= 0)
	{
		if (x1 - 1 >= 0)
			sumBottomLeft = intImg[BottomLeft];
		if (x2 >= 0)
			sumBottomRight = intImg[BottomRight];
	}
	float sum = sumBottomRight + sumTopLeft
			- sumBottomLeft - sumTopRight;

	
	
	float area = (x2 - x1 + 1) * (y2 - y1 + 1);
	return (sum / area);
}

//===========================================================================
///	CalcStaticSaliencyMap
///
/// Outputs a static saliency map with a value assigned per pixel. The values are
/// normalized in the interval [0,255] if normflag is set true (default value).
//===========================================================================
float Saliency::CalcStaticSaliencyMap(
		Mat& grey_mat,
		CvMat* salMap,
		const bool& normflag)
{
	IplImage grey_temp(grey_mat);
	IplImage * grey = &grey_temp;
	int width = grey->width;
	int height = grey->height;
	int nCounts = width * height;
	
	std::vector<float> kernel(0);
	//kernel.push_back(1.0);kernel.push_back(4.0);kernel.push_back(6.0);kernel.push_back(4.0);kernel.push_back(1.0);
	kernel.push_back(1.0);kernel.push_back(2.0);kernel.push_back(1.0);

	IplImage* smoothImg = cvCreateImage(cvGetSize(grey), IPL_DEPTH_32F, 1);
	GaussianStaticSmooth(grey, kernel, smoothImg);
	
	float* intImg = (float*)malloc(height * width * sizeof(float));
	memset(intImg, 0, height * width* sizeof(float));
	CreateStaticIntegralImage(smoothImg, intImg);
	
	
	float averageSaliency = 0;
	// calculate static saliency for each pixel
	for(int iHeight = 0; iHeight < height; iHeight++) {
		int yoff = std::min<int>(iHeight, height - iHeight);
		float* pMap = (float*)(salMap->data.ptr + iHeight * salMap->step);
		for(int iWidth = 0; iWidth < width; iWidth++) {
			int xoff = std::min<int>(iWidth, width -iWidth);
			float surround = GetStaticIntegralSum(intImg, iWidth - xoff, iHeight - yoff, iWidth + xoff, iHeight + yoff, height, width);
			yoff = std::min<int>(yoff, 0);
			xoff = std::min<int>(xoff, 0);
			float point = GetStaticIntegralSum(intImg, iWidth - xoff, iHeight - yoff, iWidth + xoff, iHeight + yoff, height, width);
			pMap[iWidth] = (surround - point) * (surround - point);
			averageSaliency += pMap[iWidth];

		}
	}
	
	if (true == normflag)
	{
		averageSaliency = Normalize(salMap, 255);
	}
	averageSaliency /= nCounts;

	free(intImg);
	cvReleaseImage(&smoothImg);
	
	return averageSaliency;
	
}



//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// motion saliency map.
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//==============================================================================
///	GaussianMotionSmooth
///
///	Blur an optical flow image with a separable binomial kernel passed in.
//==============================================================================
void Saliency::GaussianMotionSmooth(
	IplImage* inputImg,
	const std::vector<float>& kernel,
	IplImage* smoothImg)
{
	int width = inputImg->width;
	int height = inputImg->height;
	int center = kernel.size() / 2;
	IplImage* tempImg = cvCreateImage(cvGetSize(inputImg), IPL_DEPTH_32F, 2);
	
	//--------------------------------------------------------------------------
   	// Blur in the x direction.
   	//---------------------------------------------------------------------------
	for (int iHeight = 0; iHeight < height; iHeight++) {
		for (int iWidth = 0; iWidth < width; iWidth++) {
			float kernelSum(0);
			float sumX(0), sumY(0);
			float* pInput = (float*)(inputImg->imageData + iHeight * inputImg->widthStep);
			float* pTemp = (float*)(tempImg->imageData + iHeight * tempImg->widthStep);
			for (int ww = (-center); ww <= center; ww++) {
				if(((iWidth + ww) >= 0) && ((iWidth + ww) < width))
				{
					sumX += pInput[2 * (iWidth + ww)] * kernel[center + ww];
					sumY += pInput[2 * (iWidth + ww) + 1] * kernel[center + ww];
					kernelSum += kernel[center + ww];
				}
			}
			pTemp[2 * iWidth] = sumX / kernelSum;
			pTemp[2 * iWidth + 1] = sumY / kernelSum;
		}

	}

	//--------------------------------------------------------------------------
   	// Blur in the y direction.
   	//---------------------------------------------------------------------------
	for (int iHeight = 0; iHeight < height; iHeight++) {
		for (int iWidth = 0; iWidth < width; iWidth++) {
			float kernelSum(0);
			float sumX(0), sumY(0);
			float* pSmooth = (float*)(smoothImg->imageData + iHeight * smoothImg->widthStep);
			for (int hh = (-center); hh <= center; hh++) {
				if(((iHeight + hh) >= 0) && ((iHeight + hh) < height))
				{
					float* pTemp = (float*)(tempImg->imageData + (iHeight + hh) * tempImg->widthStep);
					sumX += pTemp[2 * iWidth] * kernel[center + hh];
					sumY += pTemp[2 * iWidth + 1] * kernel[center + hh];
					kernelSum += kernel[center + hh];
				}
			}
			pSmooth[2 * iWidth] = sumX / kernelSum;
			pSmooth[2 * iWidth + 1] = sumY / kernelSum;
		}
	}
	cvReleaseImage(&tempImg);
}

//==============================================================================
///	CreateMotionIntegralImage
//==============================================================================
void Saliency::CreateMotionIntegralImage(
	IplImage* inputImg,
	const DescInfo& descInfo,
	CvMat* kernelMatrix,
	float* intImg)
{
	
	int width = inputImg->width;
	int height = inputImg->height;
	int nBins = descInfo.isHof ? (descInfo.nBins - 1) : descInfo.nBins;
	float denseBase = (2 * M_PI) / float(kernelMatrix->height);
	int index = 0;

	for(int iHeight = 0; iHeight < height; iHeight++) {
		const float* f = (const float*)(inputImg->imageData + inputImg->widthStep * iHeight);
		// the histogram accumulated in the current line
		std::vector<float> sum(nBins);
		for(int iWidth = 0; iWidth < width; iWidth++, index++) {
			float shiftX = f[2 * iWidth];
			float shiftY = f[2 * iWidth + 1];
			float magnitude = sqrt(shiftX * shiftX + shiftY * shiftY);
						
			if(descInfo.isHof == 0 || magnitude > descInfo.threshold) {			
				float orientation = atan2(shiftY, shiftX);
				if (orientation < 0)
				{
					orientation += 2 * M_PI;
				}
				int iDense = static_cast<int>(roundf(orientation / denseBase));
				if (iDense >= kernelMatrix->height)
				{
					iDense = 0;
				}
				// directly apply kernel histograms
				float* ptr = (float*)(kernelMatrix->data.ptr + iDense * kernelMatrix->step);
				for (int iBin = 0; iBin < nBins; iBin++)
				{
					sum[iBin] += magnitude * ptr[iBin];
				}
			}
			
			int temp0 = index * nBins;
			if(0 == iHeight) { // for the first line
				for(int iBin = 0; iBin < nBins; iBin++)
					intImg[temp0++] = sum[iBin];
			}
			else {
				int temp1 = (index - width) * nBins;
				for(int iBin = 0; iBin < nBins; iBin++)
					intImg[temp0++] = intImg[temp1++] + sum[iBin];
			}
		}
	}
	
}

//==============================================================================
///	GetMotionIntegralSum
//==============================================================================
float Saliency::GetMotionIntegralSum(
	const float* intImg,
	int x1,
	int y1,
	int x2,
	int y2,
	const int& height,
	const int& width,
	const int& nBins,
	std::vector<float>& hist)
{
	x1 = std::max<int>(x1, 0);
	y1 = std::max<int>(y1, 0);
	x2 = std::min<int>(x2, width - 1);
	y2 = std::min<int>(y2, height - 1);
	int TopLeft = ((y1 - 1) * width + (x1 - 1)) * nBins;
	int TopRight = ((y1 - 1) * width + x2) * nBins;
	int BottomLeft = (y2 * width + (x1 - 1)) * nBins;
	int BottomRight = (y2 * width + x2) * nBins; 
	float area = (x2 - x1 + 1) * (y2 - y1 + 1);
	float sum = 0;
	for (int iBin = 0; iBin < nBins; iBin++)
	{
		float sumTopLeft(0), sumTopRight(0), sumBottomLeft(0), sumBottomRight(0);
		if (y1 - 1 >= 0) 
		{
			if (x1 - 1 >= 0)
				sumTopLeft = intImg[TopLeft + iBin];
			if (x2 >= 0)
				sumTopRight = intImg[TopRight + iBin];
		}
		if (y2 >= 0)
		{
			if (x1 - 1 >= 0)
				sumBottomLeft = intImg[BottomLeft + iBin];
			if (x2 >= 0)
				sumBottomRight = intImg[BottomRight + iBin];
		}
		hist[iBin] = sumBottomRight + sumTopLeft
					- sumBottomLeft - sumTopRight;
		hist[iBin] = std::max<int>(hist[iBin], 0);
		hist[iBin] /= area;
		sum += pow(hist[iBin], 2);
	}
	
	/*sum = sqrt(sum);
	if (sum > 0)
	{
		for (int iBin = 0; iBin < nBins; iBin++)
		{
			hist[iBin] /= sum;
		}
	}*/
	return sum;
		
}


		
//===========================================================================
///	CalcMotionSaliencyMap
///
/// Outputs a motion saliency map with a value assigned per pixel. The values are
/// normalized in the interval [0,255] if normflag is set true (default value).
//===========================================================================
float Saliency::CalcMotionSaliencyMap( 
	Mat& flow_mat, 
	const DescInfo& descInfo,
	CvMat* kernelMatrix,
	CvMat* salMap,
	const bool& normflag)
{
	IplImage flow_temp(flow_mat);
	IplImage * flow = &flow_temp;
	int width = flow->width;
	int height = flow->height;
	int nBins = descInfo.isHof ? (descInfo.nBins - 1) : descInfo.nBins;
	int nCounts = width * height;
	
	std::vector<float> kernel(0);
	//kernel.push_back(1.0);kernel.push_back(4.0);kernel.push_back(6.0);kernel.push_back(4.0);kernel.push_back(1.0);
	kernel.push_back(1.0);kernel.push_back(2.0);kernel.push_back(1.0);

	IplImage* smoothImg = cvCreateImage(cvGetSize(flow), IPL_DEPTH_32F, 2);
	GaussianMotionSmooth(flow, kernel, smoothImg);
	
	float* intImg = (float*)malloc(height * width * nBins * sizeof(float));
	memset(intImg, 0, height * width * nBins * sizeof(float));
	CreateMotionIntegralImage(smoothImg, descInfo, kernelMatrix, intImg);
	
	
	float averageSaliency = 0;
	// calculate motion saliency for each pixel
	for(int iHeight = 0; iHeight < height; iHeight++) {
		int yoff = std::min<int>(iHeight, height - iHeight);
		float* pMap = (float*)(salMap->data.ptr + iHeight * salMap->step);
		for(int iWidth = 0; iWidth < width; iWidth++) {
			int xoff = std::min<int>(iWidth, width -iWidth);
			std::vector<float> surround(nBins);
			GetMotionIntegralSum(intImg, iWidth - xoff, iHeight - yoff, iWidth + xoff, iHeight + yoff, height, width, nBins, surround);
			std::vector<float> point(nBins);
			yoff = std::min<int>(yoff, 0);
			xoff = std::min<int>(xoff, 0);
			pMap[iWidth] = 0;
			if (GetMotionIntegralSum(intImg, iWidth - xoff, iHeight - yoff, iWidth + xoff, iHeight + yoff, height, width, nBins, point) > 0)
			{
				for (int iBin = 0; iBin < nBins; iBin++)
				{
					float X1 = point[iBin];
					float X2 = surround[iBin];
					if (X1 + X2 > 0)
						pMap[iWidth] += 0.5 * pow(X1 - X2, 2) / (X1 + X2);
				}
			}
			
			averageSaliency += pMap[iWidth];

		}
	}
	
	if (true == normflag)
	{
		averageSaliency = Normalize(salMap, 255);
	}
	averageSaliency /= nCounts;

	free(intImg);
	cvReleaseImage(&smoothImg);
	
	return averageSaliency;
}
//---------------------------------end---------------------------------
