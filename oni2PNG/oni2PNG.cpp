#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <boost/program_options.hpp>
namespace po = boost::program_options;


// OpenNI
#include <XnCppWrapper.h>
#define THROW_IF_FAILED(retVal) {if (retVal != XN_STATUS_OK) throw xnGetStatusString(retVal);}

// OpenCV
#include <opencv2/opencv.hpp>

/**
 * @class
 *  Normalize colors in depth using histogram as proposed by user Vlad:
 *  http://stackoverflow.com/questions/17944590/convert-kinects-depth-to-rgb
 *  The original idea is from
 *  https://github.com/OpenNI/OpenNI2/blob/master/Samples/Common/OniSampleUtilities.h
 */

class HistogramNormalizer
{
public:
  static void run(cv::Mat& input)
  {
    std::vector<float> histogram;
    calculateHistogram(input, histogram);
    cv::MatIterator_<short> it = input.begin<short>(), it_end = input.end<short>();
    for(; it != it_end; ++it) {
      *it = histogram[*it];
    }
  }

private:
  static void calculateHistogram(const cv::Mat& depth, std::vector<float>& histogram)
  {
    int depthTypeSize = CV_ELEM_SIZE(depth.type());
    int histogramSize = pow(2., 8 * depthTypeSize);
    histogram.resize(histogramSize, 0.0f);

    unsigned int nNumberOfPoints = 0;
    cv::MatConstIterator_<short> it = depth.begin<short>(), it_end = depth.end<short>();
    for(; it != it_end; ++it) {
      if (*it != 0) {
        ++histogram[*it];
        ++nNumberOfPoints;
      }
    }

    for (int nIndex = 1; nIndex < histogramSize; ++nIndex)
    {
      histogram[nIndex] += histogram[nIndex - 1];
    }

    if (nNumberOfPoints != 0)
    {
      for (int nIndex = 1; nIndex < histogramSize; ++nIndex)
      {
        histogram[nIndex] = (256.0 * (1.0f - (histogram[nIndex] / nNumberOfPoints)));
      }
    }
  }
};




/**
 * @class
 *  Convert oni file to images
 */
class Oni2PNGConverter
{
public:
  Oni2PNGConverter()
  {

  }

  void run(const std::string& inputFile)
  {
	//common	
	xn::Context context;
	THROW_IF_FAILED(context.Init());

	xn::Player player;
	THROW_IF_FAILED(context.OpenFileRecording(inputFile.c_str(), player));
	THROW_IF_FAILED(player.SetRepeat(false));
	
	//RGB image
	xn::ImageGenerator imageGen;    
	THROW_IF_FAILED(imageGen.Create(context));
    
	XnPixelFormat pixelFormat = imageGen.GetPixelFormat();
	if (pixelFormat != XN_PIXEL_FORMAT_RGB24)
        {
		THROW_IF_FAILED(imageGen.SetPixelFormat(XN_PIXEL_FORMAT_RGB24));
	}
    
	xn::ImageMetaData xImageMap2;
	imageGen.GetMetaData(xImageMap2);
	XnUInt32 fps = xImageMap2.FPS();
	// Currently optimised for 640x480 resolution only
	std::cout<<"Depth and RGB resolution are expected to be same: 640x480 recommended"<<std::endl;
		
	XnUInt32 frame_height = xImageMap2.YRes();
	XnUInt32 frame_width = xImageMap2.XRes();
	
	//Depth
	xn::DepthGenerator depthGen;
	depthGen.Create(context);
	XnUInt32 nframes;
	player.GetNumFrames(depthGen.GetName(), nframes);
	THROW_IF_FAILED(context.StartGeneratingAll());

    	// save images
	try
	{
	        size_t iframe = 10;
	        //Extract RGB data
	        THROW_IF_FAILED(imageGen.WaitAndUpdateData());
	        xn::ImageMetaData xImageMap;
	        imageGen.GetMetaData(xImageMap);
	        XnRGB24Pixel* imgData = const_cast<XnRGB24Pixel*>(xImageMap.RGB24Data());
	        cv::Mat image(frame_height, frame_width, CV_8UC3, reinterpret_cast<void*>(imgData));

	        cv::cvtColor(image, image, CV_BGR2RGB); // opencv image format is BGR
	        std::stringstream ss_RGB;

		//display RGB image        
		cv::Mat demo;
	        image.copyTo(demo);
	        cv::namedWindow("RGB image",CV_WINDOW_AUTOSIZE);	
		cv::moveWindow("Depth image", 20,100);        
		cv::imshow("RGB image",demo);
		std::cout<<"RGB image resolution : "<<demo.cols<<"x"<<demo.rows<<std::endl;
		if(demo.cols == 640 && demo.rows == 480)
			std::cout<<"Correct RGB resolution"<<std::endl;
		else
			std::cout<<"Warning: RGB Resolution might be wrong"<<std::endl;
		cv::waitKey();

		//Extract depth data	
	        THROW_IF_FAILED(depthGen.WaitAndUpdateData());
	        xn::DepthMetaData xDepthMap;
	        depthGen.GetMetaData(xDepthMap);
	        XnDepthPixel* depthData = const_cast<XnDepthPixel*>(xDepthMap.Data());
	        cv::Mat depth(frame_height, frame_width, CV_16U, reinterpret_cast<void*>(depthData));

	        HistogramNormalizer::run(depth);
	        cv::Mat depthMat8UC1;
	        depth.convertTo(depthMat8UC1, CV_8UC1);

	        std::stringstream ss_depth;

		//display Depth image        
		cv::Mat demoD;
	        depthMat8UC1.copyTo(demoD);
		cv::namedWindow("Depth image",CV_WINDOW_AUTOSIZE);	
		cv::moveWindow("Depth image", 700,50);        
		cv::imshow("Depth image",demoD);
		std::cout<<"Depth image resolution : "<<demoD.cols<<"x"<<demoD.rows<<std::endl;
		
		if(demoD.cols == 640 && demoD.rows == 480)
			std::cout<<"Correct Depth resolution"<<std::endl;
		else
			std::cout<<"Warning: Depth Resolution might be wrong"<<std::endl;

		cv::waitKey();
		
		//store both in cache
        	ss_RGB << "./images/latest_RGB.png";
		std::string sRGBname = ss_RGB.str();
		imwrite(sRGBname,image);
	
	        ss_depth << "./images/latest_depth.png";
	        std::string sDepthName = ss_depth.str();
	  	imwrite(sDepthName, depthMat8UC1);
#if 0
		float hFOV;
		hFOV = context.hFieldOfView();
		std::cout<<"H FOV is: "<<hFOV<<std::endl;
		XnPoint3D Point2D,Point3D;
		int yy = 400;
		int xx = 600;
		cv::Mat realWorldDepth(480,640,CV_8UC1);
		
		for(int yy = 0; yy<480; yy++)
		{
			for(int xx=0; xx<640; xx++)
			{
				Point2D.X=xx;  
		           	Point2D.Y=yy;  
		           	Point2D.Z= depthData[(yy*640)+xx];  
				depthGen.ConvertProjectiveToRealWorld(1,&Point2D,&Point3D);
				realWorldDepth.at<uchar>(yy,xx) = Point3D.Z;
				if(Point3D.Z != Point2D.Z)
					std::cout<<Point2D.Z<<" "<<Point3D.Z<<std::endl;
	           	}
		}		
		cv::imshow("Real World Depth",realWorldDepth);
		cv::waitKey();
		Point2D.X=xx;  
           	Point2D.Y=yy;  
           	Point2D.Z= depthData[(yy*640)+xx];
		depthGen.ConvertProjectiveToRealWorld(1,&Point2D,&Point3D);
				  
           	std::cout<<Point2D.X<<std::endl;  
           	std::cout<<Point2D.Y<<std::endl;  
           	std::cout<<Point2D.Z<<std::endl;  
		std::cout<<Point3D.X<<std::endl;  
           	std::cout<<Point3D.Y<<std::endl;  
           	std::cout<<Point3D.Z<<std::endl;  

#endif
		cv::destroyAllWindows();
	
	}
	    catch(...)
	    {
		context.StopGeneratingAll();
		context.Release();
		throw;
		}

	context.StopGeneratingAll();
	context.Release();
  }

  
};

int main(int argc, char* argv[])
{
  try
  {
    po::options_description desc("oni2depth converts an input oni file into 2 png files - one for image and another for the depth map."
        "\n Allowed options:");
    desc.add_options()
        ("help", "produce help message")
        ("input-file", po::value< std::string >(), "input oni file");

    po::positional_options_description p;
    p.add("input-file", 1);
    
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
              options(desc).positional(p).run(), vm);

    if (vm.count("help"))
    {
      std::cout << desc << "\n";
      return 1;
    }

    
    if (!vm.count("input-file"))
      throw "input file was not set\n";

    
    
    Oni2PNGConverter converter;
    
    converter.run(vm["input-file"].as<std::string>());
      

  }
  catch (const char* error)
  {
    //  errors
    std::cout << "Error: " << error << std::endl;
    return 1;
  }
  catch (const std::exception& error)
  {
    // OpenCV exceptions are derived from std::exception
    std::cout << error.what() << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cout << "Unknown error" << std::endl;
    return 1;
  }

  return 0;
}
