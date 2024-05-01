#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <map>
#include <fstream>
#include <cmath>

#include "opencv2/opencv.hpp"
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include <ros/package.h>

#include <pcl_ros/point_cloud.h>
#include <boost/foreach.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <velodyne_pointcloud/point_types.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/common/intersections.h>


#include "lidar_camera_calibration/Utils.h"

int iteration_count = 0;//这是一个全局变量
std::vector< std::vector<cv::Point> > stored_corners;//一条直线保存四个点，一个marker共四条线，保存16个点

//scan是点云在相机坐标系下的坐标 = 激光根据初始位姿在相机坐标系下的坐标
//p是相机的投影模型，config文件中设置的初值
//num_of_markers表示一共有多少个aruco marker
//img表示图像
bool  getCorners(cv::Mat img, pcl::PointCloud<pcl::PointXYZ> scan, cv::Mat P, int num_of_markers, int MAX_ITERS)
{

	ROS_INFO_STREAM("iteration number: " << iteration_count << "\n");

	/*Masking happens here */
	cv::Mat edge_mask = cv::Mat::zeros(img.size(), CV_8UC1);
	//edge_mask(cv::Rect(520, 205, 300, 250))=1;
	edge_mask(cv::Rect(0, 0, img.cols, img.rows))=1;//创建一个图像
	img.copyTo(edge_mask, edge_mask);//作用是把mask和image重叠以后把mask中像素值为0（black）的点对应的image中的点变为透明，而保留其他点。
	//pcl::io::savePCDFileASCII ("/home/vishnu/final1.pcd", scan.point_cloud);

	img = edge_mask;

	//cv:imwrite("/home/vishnu/marker.png", edge_mask);

	pcl::PointCloud<pcl::PointXYZ> pc = scan;
	//scan = Velodyne::Velodyne(filtered_pc);
	cv::Rect frame(0, 0, img.cols, img.rows);
	
	
	//pcl::io::savePCDFileASCII("/home/vishnu/final2.pcd", scan.point_cloud);
	
	cv::Mat image_edge_laser = project(P, frame, scan, NULL);//将相机坐标系下的点云投影到当前图像中
	cv::threshold(image_edge_laser, image_edge_laser, 10, 255, 0);//将图像进行二值化操作，image_edge_laser是输入也是输出，10为阈值，255是向上最大阈值，


	

	cv::Mat combined_rgb_laser;
	std::vector<cv::Mat> rgb_laser_channels;

	rgb_laser_channels.push_back(image_edge_laser);
	rgb_laser_channels.push_back(cv::Mat::zeros(image_edge_laser.size(), CV_8UC1));
	rgb_laser_channels.push_back(img);
			 
	cv::merge(rgb_laser_channels, combined_rgb_laser);//将多个通道的图像进行合成，结果为combined_rgb_laser
	/*cv::namedWindow("combined", cv::WINDOW_NORMAL); 
	cv::imshow("combined", combined_rgb_laser);
	cv::waitKey(5);
	*/

	std::map<std::pair<int, int>, std::vector<float> > c2D_to_3D;//第一个元素是投影到图像的坐标，第二个元素是三维坐标点
	std::vector<float> point_3D;

	/* store correspondences */
	//遍历在相机坐标系下的点云坐标
	for(pcl::PointCloud<pcl::PointXYZ>::iterator pt = pc.points.begin(); pt < pc.points.end(); pt++)
	{

			// behind the camera
			if (pt->z < 0)
			{
				continue;
			}

			cv::Point xy = project(*pt, P);
			if (xy.inside(frame))
			{
				//create a map of 2D and 3D points
				point_3D.clear();
				point_3D.push_back(pt->x);
				point_3D.push_back(pt->y);
				point_3D.push_back(pt->z);
				c2D_to_3D[std::pair<int, int>(xy.x, xy.y)] = point_3D;
			}
	}

	/* print the correspondences */
	/*for(std::map<std::pair<int, int>, std::vector<float> >::iterator it=c2D_to_3D.begin(); it!=c2D_to_3D.end(); ++it)
	{
		std::cout << it->first.first << "," << it->first.second << " --> " << it->second[0] << "," <<it->second[1] << "," <<it->second[2] << "\n";
	}*/

	/* get region of interest */

	const int QUADS=num_of_markers;
	std::vector<int> LINE_SEGMENTS(QUADS, 4); //assuming each has 4 edges and 4 corners，初始化一共两个变量，都是4

	pcl::PointCloud<pcl::PointXYZ>::Ptr board_corners(new pcl::PointCloud<pcl::PointXYZ>);
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr marker(new pcl::PointCloud<pcl::PointXYZ>);
	std::vector<cv::Point3f> c_3D;//计算得到的点云在相机坐标系下的交点
	std::vector<cv::Point2f> c_2D;


	cv::namedWindow("cloud", cv::WINDOW_NORMAL);
	cv::namedWindow("polygon", cv::WINDOW_NORMAL); 
	//cv::namedWindow("combined", cv::WINDOW_NORMAL); 

	std::string pkg_loc = ros::package::getPath("lidar_camera_calibration");
	std::ofstream outfile(pkg_loc + "/conf/points.txt", std::ios_base::trunc);//打开文件，若文件已存在那么，清空文件内容.
	outfile << QUADS*4 << "\n";

	for(int q=0; q<QUADS; q++)// 遍历不同的标定板
	{
		std::cout << "---------Moving on to next marker--------\n";
		std::vector<Eigen::VectorXf> line_model;//存储的是每条线的参数
		for(int i=0; i<LINE_SEGMENTS[q]; i++)//遍历某个标定板的所有边缘线，每个标定板共有4条边缘线
		{
			cv::Point _point_;
			std::vector<cv::Point> polygon;//保存的是图像中的四个角点坐标
			int collected;

			// get markings in the first iteration only
			if(iteration_count == 0)//每调用一次getCorners函数，变量iteration_count才会被加1
			{
				polygon.clear();
				collected = 0;
				while(collected != LINE_SEGMENTS[q])
				{
					
						cv::setMouseCallback("cloud", onMouse, &_point_);//人为选的点
						
						cv::imshow("cloud", image_edge_laser);
						cv::waitKey(0);
						++collected;
						//std::cout << _point_.x << " " << _point_.y << "\n";
						polygon.push_back(_point_);
				}
				stored_corners.push_back(polygon);
			}
			
			polygon = stored_corners[4*q+i];//stored_corners这个变量非常重要只有在第一个选择时才会跟新，记录的是每张图像中用于框出来每条线的四个角点坐标

			cv::Mat polygon_image = cv::Mat::zeros(image_edge_laser.size(), CV_8UC1);
			
			rgb_laser_channels.clear();
			rgb_laser_channels.push_back(image_edge_laser);
			rgb_laser_channels.push_back(cv::Mat::zeros(image_edge_laser.size(), CV_8UC1));
			rgb_laser_channels.push_back(cv::Mat::zeros(image_edge_laser.size(), CV_8UC1));
			cv::merge(rgb_laser_channels, combined_rgb_laser);
			//在图像上绘制直线，根据我们的鼠标点击绘制得到的	
			for( int j = 0; j < 4; j++ )
			{
				cv::line(combined_rgb_laser, polygon[j], polygon[(j+1)%4], cv::Scalar(0, 255, 0));//polygon[j]直线起点，polygon[(j+1)%4]直线终点
			}

			// initialize PointClouds
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr final (new pcl::PointCloud<pcl::PointXYZ>);

			//c2D_to_3D第一个元素是投影到图像的坐标，第二个元素是三维坐标点
			for(std::map<std::pair<int, int>, std::vector<float> >::iterator it=c2D_to_3D.begin(); it!=c2D_to_3D.end(); ++it)
			{		
				if (cv::pointPolygonTest(cv::Mat(polygon), cv::Point(it->first.first, it->first.second), true) > 0)//判断这个点是否在轮廓内，这个点是投影得到的坐标点，轮廓是我们自己鼠标选择的
				{
					cloud->push_back(pcl::PointXYZ(it->second[0],it->second[1],it->second[2]));
					//3表示粗细，8表示线的类型，0表示shift
					rectangle(combined_rgb_laser, cv::Point(it->first.first, it->first.second), cv::Point(it->first.first, it->first.second), cv::Scalar(0, 0, 255), 3, 8, 0); // RED point
				}
			}
			
			if(cloud->size() < 2){ return false;}
			
			cv::imshow("polygon", combined_rgb_laser);
			cv::waitKey(4);

			//pcl::io::savePCDFileASCII("/home/vishnu/line_cloud.pcd", *cloud);
			
			

			std::vector<int> inliers;
			Eigen::VectorXf model_coefficients;


			// created RandomSampleConsensus object and compute the appropriated model
			//下面是使用pcl库的ransac，拟合出来直线
			pcl::SampleConsensusModelLine<pcl::PointXYZ>::Ptr model_l(new pcl::SampleConsensusModelLine<pcl::PointXYZ> (cloud));//点云是激光点在相机坐标系下的坐标
			//初始化ransac模型
			pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_l);
			ransac.setDistanceThreshold (0.01);
			//计算ransacn模型
			ransac.computeModel();
			//得到拟合的参数
			ransac.getInliers(inliers);
			ransac.getModelCoefficients(model_coefficients);
			line_model.push_back(model_coefficients);

			std::cout << "Line coefficients are:" << "\n" << model_coefficients << "\n";
			// copies all inliers of the model computed to another PointCloud
			pcl::copyPointCloud<pcl::PointXYZ>(*cloud, inliers, *final);
			//pcl::io::savePCDFileASCII("/home/vishnu/RANSAC_line_cloud.pcd", *final);
			*marker += *final;
		}//遍历某个标定板的所有边缘线结束


		
		/* calculate approximate intersection of lines */
		Eigen::Vector4f p1, p2, p_intersect;
		pcl::PointCloud<pcl::PointXYZ>::Ptr corners(new pcl::PointCloud<pcl::PointXYZ>);//保存的是根据四条线的交点得到的四个角点
		for(int i=0; i<LINE_SEGMENTS[q]; i++)
		{     
		/*
		输入：line_a、line_b，均为6维向量，前三维表示直线上一点的坐标，后三维表示直线的方向向量
		输出：pt1_seg、pt2_seg，最小公垂线的两个垂足，4维向量，前三维表示空间坐标，第四维是零
		*/
			pcl::lineToLineSegment(line_model[i], line_model[(i+1)%LINE_SEGMENTS[q]], p1, p2);//计算两个线的公垂线，计算交点和最小公垂线是一码事，交点即最小公垂线两个垂足的中心。结果保存在p1和p2中
			for(int j=0; j<4; j++)
			{
				p_intersect(j) = (p1(j) + p2(j))/2.0;
			}
			c_3D.push_back(cv::Point3f(p_intersect(0), p_intersect(1), p_intersect(2)));
			corners->push_back(pcl::PointXYZ(p_intersect(0), p_intersect(1), p_intersect(2)));
			std::cout << "Point of intersection is approximately: \n" << p_intersect << "\n";
			//std::cout << "Distance between the lines: " << (p1 - p2).squaredNorm () << "\n";
			std::cout << p_intersect(0) << " " << p_intersect(1) << " " << p_intersect(2) <<  "\n";
			outfile << p_intersect(0) << " " << p_intersect(1) << " " << p_intersect(2) <<  "\n";

		}
		
		*board_corners += *corners;

		std::cout << "Distance between the corners:\n";
		for(int i=0; i<4; i++)
		{
			std::cout <<  sqrt(
						  pow(c_3D[4*q+i].x - c_3D[4*q+(i+1)%4].x, 2)
						+ pow(c_3D[4*q+i].y - c_3D[4*q+(i+1)%4].y, 2)
						+ pow(c_3D[4*q+i].z - c_3D[4*q+(i+1)%4].z, 2)
						)

						<< std::endl;
		}


	}
	outfile.close();

	iteration_count++;
	if(iteration_count == MAX_ITERS)
	{
		ros::shutdown();
	}
	return true;
	/* store point cloud with intersection points */
	//pcl::io::savePCDFileASCII("/home/vishnu/RANSAC_marker.pcd", *marker);
	//pcl::io::savePCDFileASCII("/home/vishnu/RANSAC_corners.pcd", *board_corners);
}
