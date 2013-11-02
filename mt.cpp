#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);
void showOriginal(Mat frame);
void drawOsd(Mat frame);

Mat previous_frame;
//Mat prev;
String window_motion = "Motion";

/** @function main */
int main(void) {
  VideoCapture capture;
  Mat frame;

  capture.open(-1);
  if (!capture.isOpened()){
    printf("--(!)Error opening video capture\n");
    return -1;
  }

  /// Create a motion window
  namedWindow(window_motion, WINDOW_AUTOSIZE);

  while (capture.read(frame)){
    if(frame.empty()){
      printf(" --(!) No captured frame -- Break!");
      break;
    }
    
    // Show the original stream
    imshow("Live Feed", frame);

    detectAndDisplay(frame);

    int c = waitKey(10);
    // Esc
    if((char)c == 27){
      break;
    }
  }
  
  return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame) {
  Mat grey, diff, blurred, canny, result, eroded, bw;
  int blur_size = 12;
  int tresh = 3;

  // Initialize the previous frame when function is called for the first time.  
  if(previous_frame.empty()){
    frame.copyTo(previous_frame);
  }

  // Difference to last frame
  absdiff(previous_frame, frame, diff);
  // Convert to greyscale, since this removes a lot of webcam noise
  cvtColor(diff, grey, COLOR_BGR2GRAY);
  // Blur the frame to filter some noise
  blur(grey, blurred, Size(blur_size, blur_size));

  //imshow("blurred", blurred);
  //imshow("grey", grey);

  Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3), Point(1, 1));
  /// Apply the erosion operation
  erode(blurred, eroded, element);

  imshow("Erroded before", eroded);

  //addWeighted(prev, 0.7, eroded, 1, 0, eroded);

  threshold(eroded, bw, 7, 255, 0);

  imshow("Erroded after", bw);

  // Edge filter
  Canny(bw, canny, tresh, tresh*3, 3);

  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  findContours(bw, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

  vector<vector<Point> > contours_poly( contours.size() );
  vector<Rect> boundRect( contours.size() );

  int minX = -1;
  int minY = -1;
  int maxX = -1;
  int maxY = -1;
  
  for(int i = 0; i < contours.size(); i++){
    approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
    boundRect[i] = boundingRect( Mat(contours_poly[i]) );
    if(minX < 0 ) {
      minX = boundRect[i].tl().x;
      minY = boundRect[i].br().y;
      maxX = boundRect[i].tl().x;
      maxY = boundRect[i].br().y;    
    }

    if(boundRect[i].tl().x < minX) minX = boundRect[i].tl().x;
    if(boundRect[i].tl().y < minY) minY = boundRect[i].tl().y;

    if(boundRect[i].br().x > maxX) maxX = boundRect[i].br().x;
    if(boundRect[i].br().y > maxY) maxY = boundRect[i].br().y;
  }
  cvtColor(canny, result, COLOR_GRAY2BGR);

  // Draw bounding rect and center
  if(minX > -1){
    rectangle( result, Point(minX, minY), Point(maxX, maxY), Scalar(0, 255, 0), 2, 8, 0 );
    Point center(minX + (maxX - minX)/2, minY + (maxY - minY)/2);
    circle(result, center, 1, Scalar(0, 0, 255), 3, 8, 0);
  }

  drawOsd(result);

  imshow(window_motion, result);
  frame.copyTo(previous_frame);
}

void drawOsd(Mat frame) {

  // Draw a crosshair
  Point center(frame.cols/2, frame.rows/2);
  circle(frame, center, 7, Scalar(0, 0, 255),1, 8, 0);
  // Vertical middle line
  Point start =  Point(frame.cols/2, 0);
  Point end =  Point(frame.cols/2, frame.rows-1);
  line(frame, start, end, Scalar(0, 0, 255), 1, 8);
  // Horizonzal middle line
  start =  Point(0, frame.rows/2);
  end =  Point(frame.cols-1, frame.rows/2);
  line(frame, start, end, Scalar(0, 0, 255), 1, 8);
}