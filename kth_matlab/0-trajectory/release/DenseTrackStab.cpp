#include "DenseTrackStab.h"
#include "Initialize.h"
#include "Descriptors.h"
#include "OpticalFlow.h"
#include "Saliency.h"
#include <iostream>
#include <ctime>
#include <time.h>

using namespace cv;
using namespace std;

int show_track = 0; // set show_track = 1, if you want to visualize the trajectories
float threshold_s = 1.5;

int main(int argc, char** argv)
{

	VideoCapture capture;
	char* video = argv[1];
	int flag = arg_parse(argc, argv);
	capture.open(video);

	if(!capture.isOpened()) {
		fprintf(stderr, "Could not initialize capturing..\n");
		return -1;
	}

	clock_t ttt;
	double avg_time=0;

	int frame_num = 0;
	TrackInfo trackInfo;
	DescInfo hogInfo, hofInfo, mbhInfo;

	/**-****************************/
	Saliency sal;
	float staticRatio = 0.5;
	float dynamicRatio = 0.5;
	// kernel matrix of densely sampled angles
	CvMat* kernelMatrix = cvCreateMat(3600, 8, CV_32FC1); 


	InitTrackInfo(&trackInfo, track_length, init_gap);
	InitDescInfo(&hogInfo, 8, false, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&hofInfo, 9, true, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&mbhInfo, 8, false, patch_size, nxy_cell, nt_cell);

	InitKernelMatrix(kernelMatrix, 5);

	SeqInfo seqInfo;
	InitSeqInfo(&seqInfo, video);

	
	if(flag)
		seqInfo.length = end_frame - start_frame + 1;

//	fprintf(stderr, "video size, length: %d, width: %d, height: %d\n", seqInfo.length, seqInfo.width, seqInfo.height);

	if(show_track == 1)
		namedWindow("DenseTrackStab", 0);

	SurfFeatureDetector detector_surf(200);
	SurfDescriptorExtractor extractor_surf(true, true);

	std::vector<Point2f> prev_pts_flow, pts_flow;
	std::vector<Point2f> prev_pts_surf, pts_surf;
	std::vector<Point2f> prev_pts_all, pts_all;

	std::vector<KeyPoint> prev_kpts_surf, kpts_surf;
	Mat prev_desc_surf, desc_surf;
	Mat flow, human_mask;

	Mat image, prev_grey, grey;

	std::vector<float> fscales(0);
	std::vector<Size> sizes(0);

	std::vector<Mat> prev_grey_pyr(0), grey_pyr(0), flow_pyr(0), flow_warp_pyr(0);
	std::vector<Mat> prev_poly_pyr(0), poly_pyr(0), poly_warp_pyr(0);

	std::vector<std::list<Track> > xyScaleTracks;
	int init_counter = 0; // indicate when to detect new feature points
	while(true)
	{
		Mat frame;
		int i, j, c;

		// get a new frame
		capture >> frame;
		if(frame.empty())
			break;

		if(frame_num < start_frame || frame_num > end_frame)
		{
			frame_num++;
			continue;
		}
		if(frame_num == start_frame)
		{
			//cout<<" ==================================the first frame"<<endl;
			image.create(frame.size(), CV_8UC3);
			grey.create(frame.size(), CV_8UC1);
			prev_grey.create(frame.size(), CV_8UC1);

			InitPry(frame, fscales, sizes);

			BuildPry(sizes, CV_8UC1, prev_grey_pyr);
			BuildPry(sizes, CV_8UC1, grey_pyr);
			BuildPry(sizes, CV_32FC2, flow_pyr);
			BuildPry(sizes, CV_32FC2, flow_warp_pyr);

			BuildPry(sizes, CV_32FC(5), prev_poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_warp_pyr);

			xyScaleTracks.resize(scale_num);

			frame.copyTo(image);
			cvtColor(image, prev_grey, CV_BGR2GRAY);

			for(int iScale = 0; iScale < scale_num; iScale++) {
				if(iScale == 0)
					prev_grey.copyTo(prev_grey_pyr[0]);
				else
					resize(prev_grey_pyr[iScale-1], prev_grey_pyr[iScale], prev_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

				// dense sampling feature points
				std::vector<Point2f> points(0);
				DenseSample(prev_grey_pyr[iScale], points, quality, min_distance);

				// save the feature points
				std::list<Track>& tracks = xyScaleTracks[iScale];
				for(i = 0; i < points.size(); i++)
					tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
			}

			// compute polynomial expansion
			my::FarnebackPolyExpPyr(prev_grey, prev_poly_pyr, fscales, 7, 1.5);

			human_mask = Mat::ones(frame.size(), CV_8UC1);
			

			detector_surf.detect(prev_grey, prev_kpts_surf, human_mask);
			extractor_surf.compute(prev_grey, prev_kpts_surf, prev_desc_surf);

			frame_num++;
			continue;
		}
		//cout<<" ==================================the second frame"<<frame_num<<"/"<<seqInfo.length<<endl;
		init_counter++;
		frame.copyTo(image);
		cvtColor(image, grey, CV_BGR2GRAY);

		// match surf features
		
		detector_surf.detect(grey, kpts_surf, human_mask);
		extractor_surf.compute(grey, kpts_surf, desc_surf);
		ComputeMatch(prev_kpts_surf, kpts_surf, prev_desc_surf, desc_surf, prev_pts_surf, pts_surf);

		// compute optical flow for all scales once
		my::FarnebackPolyExpPyr(grey, poly_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_pyr, flow_pyr, 10, 2);

		MatchFromFlow(prev_grey, flow_pyr[0], prev_pts_flow, pts_flow, human_mask);
		MergeMatch(prev_pts_flow, pts_flow, prev_pts_surf, pts_surf, prev_pts_all, pts_all);

		Mat H = Mat::eye(3, 3, CV_64FC1);
		if(pts_all.size() > 50)
		{
			std::vector<unsigned char> match_mask;
			Mat temp = findHomography(prev_pts_all, pts_all, RANSAC, 1, match_mask);
			if(countNonZero(Mat(match_mask)) > 25)
				H = temp;
		}

		Mat H_inv = H.inv();
		Mat grey_warp = Mat::zeros(grey.size(), CV_8UC1);
		MyWarpPerspective(prev_grey, grey, grey_warp, H_inv); // warp the second frame

		// compute optical flow for all scales once
		my::FarnebackPolyExpPyr(grey_warp, poly_warp_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_warp_pyr, flow_warp_pyr, 10, 2);

		for(int iScale = 0; iScale < scale_num; iScale++)
		{
			/*===========================================================*/
			std::vector<CvPoint2D32f> points_in(0);
			std::list<Track>& tracks = xyScaleTracks[iScale];
			for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); ++iTrack)
			{
				points_in.push_back(iTrack->point[iTrack->index]);
			}
			int count = points_in.size();
			/*===========================================================*/


			if(iScale == 0)
				grey.copyTo(grey_pyr[0]);
			else
				resize(grey_pyr[iScale-1], grey_pyr[iScale], grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

			int width = grey_pyr[iScale].cols;
			int height = grey_pyr[iScale].rows;

			// compute the integral histograms
			DescMat* hogMat = InitDescMat(height+1, width+1, hogInfo.nBins);
			HogComp(prev_grey_pyr[iScale], hogMat->desc, hogInfo,kernelMatrix);

			DescMat* hofMat = InitDescMat(height+1, width+1, hofInfo.nBins);
			HofComp(flow_warp_pyr[iScale], hofMat->desc, hofInfo,kernelMatrix);

			DescMat* mbhMatX = InitDescMat(height+1, width+1, mbhInfo.nBins);
			DescMat* mbhMatY = InitDescMat(height+1, width+1, mbhInfo.nBins);
			MbhComp(flow_warp_pyr[iScale], mbhMatX->desc, mbhMatY->desc, mbhInfo,kernelMatrix);

			//cout<<" calculate static saliency map " <<endl;
			CvMat* staticSalMap = cvCreateMat(height, width, CV_32FC1);
			float staticSaliency = sal.CalcStaticSaliencyMap(prev_grey_pyr[iScale], staticSalMap, true);
			if( show_track == 1 ) 
			{
				IplImage* pImg = cvCreateImage(cvGetSize(staticSalMap), IPL_DEPTH_32F, 1);
				cvConvert(staticSalMap, pImg);
				char str[20];
				sprintf(str, "saliency1/%d.jpg", frame_num);
				if (iScale == 0)
					cvSaveImage(str, pImg);
				cvReleaseImage( &pImg );
			}
			//cout<<" calculate dynamic saliency map " <<endl;
			// calculate dynamic saliency map
			CvMat* dynamicSalMap = cvCreateMat(height, width, CV_32FC1);
			float dynamicSaliency = sal.CalcMotionSaliencyMap(flow_warp_pyr[iScale], hofInfo, kernelMatrix, dynamicSalMap, true);
			if( show_track == 1 ) 
			{
				IplImage* pImg = cvCreateImage(cvGetSize(dynamicSalMap), IPL_DEPTH_32F, 1);
				cvConvert(dynamicSalMap, pImg);
				char str[20];
				sprintf(str, "saliency2/%d.jpg", frame_num);
				if (iScale == 0)
					cvSaveImage(str, pImg);
				cvReleaseImage( &pImg );
			}
			
			// calculate combined saliency map
			CvMat* salMap = cvCreateMat(height, width, CV_32FC1);
			for(int iHeight = 0; iHeight < height; iHeight++) 
			{
				float* pMap = (float*)(salMap->data.ptr + iHeight * salMap->step);
				float* pStaticMap = (float*)(staticSalMap->data.ptr + iHeight * staticSalMap->step);
				float* pDynamicMap = (float*)(dynamicSalMap->data.ptr + iHeight * dynamicSalMap->step);
				for(int iWidth = 0; iWidth < width; iWidth++) {
					pMap[iWidth] = staticRatio * pStaticMap[iWidth] + dynamicRatio * pDynamicMap[iWidth];
				}
			}

			/*=======================================================================*/
			float averageSaliency = staticRatio * staticSaliency + dynamicRatio * dynamicSaliency;
			
			std::vector<int> status(count);
			std::vector<CvPoint2D32f> points_out(count);
			std::vector<float> saliency(count);
			//cout<<count<<" is total number of feature"<<endl;
			// track feature points by median filtering
			//cout<<" release  saliency map " <<endl;
			OpticalFlowTracker(flow_warp_pyr[iScale], salMap, points_in, points_out, status, saliency);
			cvReleaseMat(&salMap);
			cvReleaseMat(&dynamicSalMap);
			cvReleaseMat(&staticSalMap);

			/*=======================================================================*/
			j = 0;
			// track feature points in each scale separately
			//std::list<Track>&  = tracks = xyScaleTracks[iScale];
			tracks = xyScaleTracks[iScale];
			//cout<<"tracks size is "<<tracks.size()<<endl;
			j = 0;
			for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();j++)
			{
				if( status[j] == 1 ){
						int index = iTrack->index;
						Point2f prev_point = iTrack->point[index];
						int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0), width-1);
						int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0), height-1);

						Point2f point;
						point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2*x];
						point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2*x+1];
	 
						if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
							iTrack = tracks.erase(iTrack);
							continue;
						} //bug occured before.
						iTrack->disp[index].x = flow_warp_pyr[iScale].ptr<float>(y)[2*x];
						iTrack->disp[index].y = flow_warp_pyr[iScale].ptr<float>(y)[2*x+1];
						iTrack->saliency[index] = saliency[j];
						iTrack->averageSaliency[index] = averageSaliency;
						// get the descriptors for the feature point
						RectInfo rect;
						GetRect(prev_point, rect, width, height, hogInfo);
						GetDesc(hogMat, rect, hogInfo, iTrack->hog, index);
						GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
						GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
						GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
						iTrack->addPoint(point);
						++iTrack;
				}
				else{
					iTrack = tracks.erase(iTrack);
				}
			}
			ReleDescMat(hogMat);
			ReleDescMat(hofMat);
			ReleDescMat(mbhMatX);
			ReleDescMat(mbhMatY);
		}

		for( int iScale = 0; iScale < scale_num; ++iScale ) 
		{
			std::list<Track>& tracks = xyScaleTracks[iScale]; // output the features for each scale
			for( std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); )
			{
				// if the trajectory achieves the maximal length
				if(iTrack->index >= trackInfo.length) {
					std::vector<Point2f> trajectory(trackInfo.length+1);
					std::vector<float> saliency(trackInfo.length+1);
					std::vector<float> averageSaliency(trackInfo.length+1);

					for(int i = 0; i <= trackInfo.length; ++i){
						trajectory[i] = iTrack->point[i]*fscales[iScale];
						saliency[i] = iTrack->saliency[i];
						averageSaliency[i] = iTrack->averageSaliency[i];
					}
				
					std::vector<Point2f> displacement(trackInfo.length);
					for (int i = 0; i < trackInfo.length; ++i)
						displacement[i] = iTrack->disp[i]*fscales[iScale];
	
					float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);

					if(( IsValid(trajectory, mean_x, mean_y, var_x, var_y, length,saliency, averageSaliency, threshold_s) == 1 ) && IsCameraMotion(displacement))
					{
						//output the trajectory
						printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t", frame_num, mean_x, mean_y, var_x, var_y, length, fscales[iScale]);

						// for spatio-temporal pyramid
						printf("%f\t", std::min<float>(std::max<float>(mean_x/float(seqInfo.width), 0), 0.999));
						printf("%f\t", std::min<float>(std::max<float>(mean_y/float(seqInfo.height), 0), 0.999));
						printf("%f\t", std::min<float>(std::max<float>((frame_num - trackInfo.length/2.0 - start_frame)/float(seqInfo.length), 0), 0.999));
					
						// output the trajectory
						for (int i = 0; i < trackInfo.length; ++i)
							printf("%f\t%f\t", displacement[i].x, displacement[i].y);
		
						PrintDesc(iTrack->hog, hogInfo, trackInfo);
						PrintDesc(iTrack->hof, hofInfo, trackInfo);
						PrintDesc(iTrack->mbhX, mbhInfo, trackInfo);
						PrintDesc(iTrack->mbhY, mbhInfo, trackInfo);
						printf("\n");
						

						if( show_track == 1 )
						{
							DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image);
						}
					}
					iTrack = tracks.erase(iTrack);
				}
				else
					++iTrack;
			}
		}


		if(init_counter != trackInfo.gap)
			continue;
		// detect new feature points every gap frames
		for (int iScale = 0; iScale < scale_num; ++iScale) {
			std::list<Track>& tracks = xyScaleTracks[iScale];
			std::vector<Point2f> points(0);
			for(std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++)
				points.push_back(iTrack->point[iTrack->index]);

			DenseSample(grey_pyr[iScale], points, quality, min_distance);
			// save the new feature points
			for(i = 0; i < points.size(); i++)
				tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
		}

		init_counter = 0;
		grey.copyTo(prev_grey);
		for(i = 0; i < scale_num; i++) {
			grey_pyr[i].copyTo(prev_grey_pyr[i]);
			poly_pyr[i].copyTo(prev_poly_pyr[i]);
		}

		prev_kpts_surf = kpts_surf;
		desc_surf.copyTo(prev_desc_surf);

		frame_num++;

		if( show_track == 1 ) {
			imshow( "DenseTrackStab", image);
			c = cvWaitKey(3);
			if((char)c == 27) break;
		}
	}

	if( show_track == 1 )
		destroyWindow("DenseTrackStab");

	ttt=clock()-ttt;

	float process_time=(float)ttt/CLOCKS_PER_SEC;
	avg_time+=process_time;
	cvReleaseMat(&kernelMatrix);
	//cout<<"average_time: "<<avg_time<<endl;

	return 0;
}
