/**
 * @brief: Monocular 3D Reconstruction
*/


#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

#include <boost/timer.hpp>
#define FMT_HEADER_ONLY
#include <fmt/format.h>
// for sophus
#include <sophus/se3.hpp>


// for eigen
#include <Eigen/Core>
#include <Eigen/Geometry>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


//Headers



//Camera Intrinsics
const int border = 20;         
const int width = 640;          
const int height = 480;         
const double fx = 481.2f;       
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;

//Epipolar Params
//Block size
const int ncc_window_size = 3;    
//Number of pixel inside the block
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); 
//Uncertainty Parameters for the sensor fusion update
const double min_cov = 0.1;     
const double max_cov = 10;      

//Dataset read I/O operations
bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    vector<Sophus::SE3d> &poses,
    cv::Mat &ref_depth
);

/**
 * Update Equation based on new data
 * @param ref           Base Image
 * @param curr          Current image
 * @param R_T_C         Pose of current wrt Base 
 * @param depth         depth 
 * @param depth_cov     variance
 */
bool update(
    const cv::Mat &ref,
    const cv::Mat &curr,
    const Sophus::SE3d &R_T_c,
    cv::Mat &depth,
    cv::Mat &depth_cov2
);

/**
 * Search along epipolar line
 * @param ref           Reference image
 * @param curr          Current image
 * @param R_T_C         Pose of current frame wrt Base
 * @param pt_ref        Location of point in Base Frame p1
 * @param depth_mu      Current state of depth
 * @param depth_cov     Covariance
 * @param pt_curr       Point in Current image p2
 * @param epipolar_direction  Epipolar Direction
 */
bool epipolarSearch(
    const cv::Mat &ref,
    const cv::Mat &curr,
    const Sophus::SE3d &R_T_C,
    const Eigen::Vector2d &pt_ref,
    const double &depth_mu,
    const double &depth_cov,
    Eigen::Vector2d &pt_curr,
    Eigen::Vector2d &epipolar_direction
);

/**
 * Update Depth Filter
 * @param pt_ref    p1
 * @param pt_curr   p2
 * @param R_T_C     Pose of p2 wrt p1
 * @param epipolar_direction epipolar direction
 * @param depth     State Vector
 * @param depth_cov2    Covariance
 * @return          
 */
bool updateDepthFilter(
    const Eigen::Vector2d &pt_ref,
    const Eigen::Vector2d &pt_curr,
    const Sophus::SE3d &R_T_C,
    const Eigen::Vector2d &epipolar_direction,
    cv::Mat &depth,
    cv::Mat &depth_cov2
);

/**
 * Calculate NCC Score
 * @param ref       Reference image
 * @param curr      Current Image
 * @param pt_ref    p1
 * @param pt_curr   p2
 * @return          NCC Score
 */
double NCC(const cv::Mat &ref, const cv::Mat &curr, const Eigen::Vector2d &pt_ref, const Eigen::Vector2d &pt_curr);

// Bilinear gray scale interpolation - for subpixel depth accuracy
inline double getBilinearInterpolatedValue(const cv::Mat &img, const Eigen::Vector2d &pt) {
    //Pointer to the start of the image
    uchar *d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
    //Calculate Fractional part of the coordinates
    double xx = pt(0, 0) - floor(pt(0, 0));
    double yy = pt(1, 0) - floor(pt(1, 0));
    //Bilinear interpolation calculation
    return ((1 - xx) * (1 - yy) * double(d[0]) +
            xx * (1 - yy) * double(d[1]) +
            (1 - xx) * yy * double(d[img.step]) +
            xx * yy * double(d[img.step + 1])) / 255.0;
}

//Does what it says
void plotDepth(const cv::Mat &depth_truth, const cv::Mat &depth_estimate);

//Converts pixel to Calibrated camera coordinates (without pose)
inline Eigen::Vector3d px2cam(const Eigen::Vector2d px) {
    return Eigen::Vector3d(
        (px(0, 0) - cx) / fx,
        (px(1, 0) - cy) / fy,
        1
    );
}

// Converts Calibrated Camera coordinates to pixel
inline Eigen::Vector2d cam2px(const Eigen::Vector3d p_cam) {
    return Eigen::Vector2d(
        p_cam(0, 0) * fx / p_cam(2, 0) + cx,
        p_cam(1, 0) * fy / p_cam(2, 0) + cy
    );
}

//Checks if the point is inside the image or not. 
inline bool inside(const Eigen::Vector2d &pt) {
    return pt(0, 0) >= border && pt(1, 0) >= border
           && pt(0, 0) + border < width && pt(1, 0) + border <= height;
}

// Shows the matched point
void showEpipolarMatch(const cv::Mat &ref, const cv::Mat &curr, const Eigen::Vector2d &px_ref, const Eigen::Vector2d &px_curr);

// Shows the epipolar line
void showEpipolarLine(const cv::Mat &ref, const cv::Mat &curr, const Eigen::Vector2d &px_ref, const Eigen::Vector2d &px_min_curr,
                      const Eigen::Vector2d &px_max_curr);

/// Evaluate the estimated depth with the ground truth - for performance
void evaludateDepth(const cv::Mat &depth_truth, const cv::Mat &depth_estimate);
// ------------------------------------------------------------------




int main(void){
    //Reading the dataset
    string dataset = "/home/dhruv/codes/dense-reconstruction/test_data";
    vector<string> color_image_files;
    vector<Sophus::SE3d> poses_TWC;
    cv::Mat ref_depth;
    if(!readDatasetFiles(dataset, color_image_files, poses_TWC, ref_depth)) cout << "Error in reading dataset" ;

    cout << "Read " << color_image_files.size() << " images" << endl;

    //Take the Base frame
    cv::Mat ref = cv::imread(color_image_files[0], 0);
    
    Sophus::SE3d pose_ref_TWC = poses_TWC[0];
    double init_depth = 3.0; //initializing state vector
    double init_cov2 = 3.0;

    //Depth and covariance map of the entire image
    cv::Mat depth(height, width, CV_64F, init_depth);             
    cv::Mat depth_cov2(height, width, CV_64F, init_cov2);         
    
    //Loop
    for (size_t index = 1; index < color_image_files.size(); index++) {
        if(index%2==0) continue;
        cout << "#";
        cv::Mat curr = cv::imread(color_image_files[index], 0); //Read the next image
        if (curr.data == nullptr) continue; //If data is shit, continue
        //Read the next pose
        Sophus::SE3d pose_curr_TWC = poses_TWC[index];
        //Calculate the relative pose wrt base frame
        Sophus::SE3d R_T_C = pose_curr_TWC.inverse() * pose_ref_TWC;   // T_C_W * T_W_R = T_C_R
        update(ref, curr, R_T_C, depth, depth_cov2); //Update the depth
        evaludateDepth(ref_depth, depth); //evaluate wrt gt

        cv::imwrite(to_string(index)+".png", depth);
        // if (index>160){
        //     plotDepth(ref_depth, depth); //Plot the function
        //     cv::waitKey(1);
        // }
        
        

    }
    cv::waitKey(0);


    cout << "estimation returns, saving depth map ..." << endl;
    cv::imwrite("depth.png", depth);
    cout << "done." << endl;

    return 1;
}

//Dataset I/O - dont care what it does
bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    std::vector<Sophus::SE3d> &poses,
    cv::Mat &ref_depth) {
    ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin) return false;

    while (!fin.eof()) {
        string image;
        fin >> image;
        double data[7];
        for (double &d:data) fin >> d;

        color_image_files.push_back(path + string("/images/") + image);
        poses.push_back(
            Sophus::SE3d(Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                 Eigen::Vector3d(data[0], data[1], data[2]))
        );
        if (!fin.good()) break;
    }
    fin.close();

    // load reference depth
    fin.open(path + "/depthmaps/scene_000.depth");
    ref_depth = cv::Mat(height, width, CV_64F);
    if (!fin) return false;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            double depth = 0;
            fin >> depth;
            ref_depth.ptr<double>(y)[x] = depth / 100.0;
        }

    return true;
}



//Min covariance Update - Kalman equation
bool update(const cv::Mat &ref, const cv::Mat &curr, const Sophus::SE3d &T_C_R, cv::Mat &depth, cv::Mat &depth_cov2) {
    for (size_t x = border; x < width - border; x++)
        for (size_t y = border; y < height - border; y++) {
            //If converged or diverged
            if (depth_cov2.ptr<double>(y)[x] < min_cov || depth_cov2.ptr<double>(y)[x] > max_cov) 
                continue;

            //Perform epipolar search
            Eigen::Vector2d pt_curr; //Pixel in current image
            Eigen::Vector2d epipolar_direction; //Epipolar direction  - interesting
            //Get the depth for that pixel - We get pixel in current image (pt_curr)
            bool ret = epipolarSearch(
                ref,
                curr,
                T_C_R,
                Eigen::Vector2d(x, y),
                depth.ptr<double>(y)[x],
                sqrt(depth_cov2.ptr<double>(y)[x]),
                pt_curr,
                epipolar_direction
            );

            if (ret == false) // Shit depth
                continue;

            // if (x>600 && y>455)
            // {
            //     // showEpipolarMatch(ref, curr, Vector2d(x, y), pt_curr);
            // }

            //Update the depth
            updateDepthFilter(Eigen::Vector2d(x, y), pt_curr, T_C_R, epipolar_direction, depth, depth_cov2);
        }
        return 0;
}

//Search the depth
bool epipolarSearch(
    const cv::Mat &ref, const cv::Mat &curr,
    const Sophus::SE3d &R_T_C, const Eigen::Vector2d &pt_ref,
    const double &depth_mu, const double &depth_cov,
    Eigen::Vector2d &pt_curr, Eigen::Vector2d &epipolar_direction){
    //THe below process is just reprojecting 3D Pw

    //Convert pixel to camera calibrated coordinates (intr^-1)@Pc
    Eigen::Vector3d f_ref = px2cam(pt_ref);
    //Normalize
    f_ref.normalize();
    //Depth in cam1 frame = camera_calibrated_coordinates*depth (intr^-1)@Pc*depth
    Eigen::Vector3d P_ref = f_ref * depth_mu;    
    //inv(extr)(inv(intr))*Pc*depth = Pc -> Reprojected with the estimated depth
    //This gives predicted pixel px_mean_curr where the reference pixel might appear in the current frame
    Eigen::Vector2d px_mean_curr = cam2px(R_T_C * P_ref); // 按深度均值投影的像素
    /////////////////
    //Minimum and maximum depth = +/- 3 sigma. Basically the new depth should not exceeed covariances
    double d_min = depth_mu - 3 * depth_cov, d_max = depth_mu + 3 * depth_cov;
    //Min Depth not be insane
    if (d_min < 0.1) d_min = 0.1;

    //The 3D position P_ref is projected onto the current image using the transformation R_T_C.
    //Basically we are converting 2D covariance to 3D depth range
    Eigen::Vector2d px_min_curr = cam2px(R_T_C * (f_ref * d_min));    
    Eigen::Vector2d px_max_curr = cam2px(R_T_C * (f_ref * d_max));    

    //Epipolar Line segment
    Eigen::Vector2d epipolar_line = px_max_curr - px_min_curr;    
    //This is same as direction
    epipolar_direction = epipolar_line;       
    epipolar_direction.normalize();

    //Mid point of ep line
    double half_length = 0.5 * epipolar_line.norm();    
    if (half_length > 100) half_length = 100;   

    //[debug] Would like to see how it looks
    showEpipolarLine( ref, curr, pt_ref, px_min_curr, px_max_curr );

    //NCC
    double best_ncc = -1.0;
    Eigen::Vector2d best_px_curr;
    //Step size is the diagonal of the pixel square/2. 
    for (double l = -half_length; l <= half_length; l += 0.7) { 
        //px_mean_curr -> predicted location
        Eigen::Vector2d px_curr = px_mean_curr + l * epipolar_direction;  
        //Range checks
        if (!inside(px_curr))
            continue;
        //Calculate NCC for the block
        double ncc = NCC(ref, curr, pt_ref, px_curr);
        if (ncc > best_ncc) {
            best_ncc = ncc;
            best_px_curr = px_curr;
        }
    }
    //If NCC is too bad -> ignore the pixel
    if (best_ncc < 0.85f)      
        return false;
    //Optimal Pixel    
    pt_curr = best_px_curr;
    return true;
}

//NCC Function
double NCC(
    const cv::Mat &ref, const cv::Mat &curr,
    const Eigen::Vector2d &pt_ref, const Eigen::Vector2d &pt_curr) {
    double mean_ref = 0, mean_curr = 0;
    vector<double> values_ref, values_curr; 
    for (int x = -ncc_window_size; x <= ncc_window_size; x++)
        for (int y = -ncc_window_size; y <= ncc_window_size; y++) {
            double value_ref = (double)(ref.ptr<uchar>(int(y + pt_ref(1, 0)))[int(x + pt_ref(0, 0))]) / 255.0;
            mean_ref += value_ref;
            double value_curr = getBilinearInterpolatedValue(curr, pt_curr + Eigen::Vector2d(x, y));
            mean_curr += value_curr;
            values_ref.push_back(value_ref);
            values_curr.push_back(value_curr);
        }

    mean_ref /= ncc_area;
    mean_curr /= ncc_area;

    double numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for (int i = 0; i < values_ref.size(); i++) {
        double n = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr);
        numerator += n;
        demoniator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);
        demoniator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr);
    }
    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);   
}

template<typename _Matrix_Type_>
_Matrix_Type_ pseudoInverse(const _Matrix_Type_ &a, double epsilon = std::numeric_limits<double>::epsilon())
{

	Eigen::JacobiSVD< _Matrix_Type_ > svd(a ,Eigen::ComputeFullU | Eigen::ComputeFullV);
        // For a non-square matrix
        // Eigen::JacobiSVD< _Matrix_Type_ > svd(a ,Eigen::ComputeThinU | Eigen::ComputeThinV);
	double tolerance = epsilon * std::max(a.cols(), a.rows()) *svd.singularValues().array().abs()(0);
	return svd.matrixV() *  (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
}

bool updateDepthFilter(
    const Eigen::Vector2d &pt_ref,
    const Eigen::Vector2d &pt_curr,
    const Sophus::SE3d &T_C_R,
    const Eigen::Vector2d &epipolar_direction,
    cv::Mat &depth,
    cv::Mat &depth_cov2) {
    //Transformation from Reference to Current Frame
    Sophus::SE3d C_T_R = T_C_R.inverse();
    //Calibrated Camera Coordinates of reference frame pixels
    Eigen::Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    //Calibrated Camera Coordinates of Current frame pixels (obtained from NCC search over epipolar line)
    Eigen::Vector3d f_curr = px2cam(pt_curr);
    f_curr.normalize();

    //lambda q = mu*Rp + t
    //[q - Rp ][d1;d2] = t
    // Try with psuedo inverse approach
    // pseudoInverse()

    Eigen::Vector3d t = C_T_R.translation();
    Eigen::Vector3d f2 = C_T_R.so3() * f_curr;
    Eigen::Vector2d b = Eigen::Vector2d(t.dot(f_ref), t.dot(f2));
    Eigen::Matrix2d A;
    A(0, 0) = f_ref.dot(f_ref);
    A(0, 1) = -f_ref.dot(f2);
    A(1, 0) = -A(0, 1);
    A(1, 1) = -f2.dot(f2);
    Eigen::Vector2d ans = A.inverse() * b;
    Eigen::Vector3d xm = ans[0] * f_ref;           // ref 
    Eigen::Vector3d xn = t + ans[1] * f2;          // cur 
    Eigen::Vector3d p_esti = (xm + xn) / 2.0;      
    double depth_estimation = p_esti.norm();    

    //Updating Covariance of depth using Kalman Formula
    Eigen::Vector3d p = f_ref * depth_estimation;
    Eigen::Vector3d a = p - t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f_ref.dot(t) / t_norm);
    double beta = acos(-a.dot(t) / (a_norm * t_norm));
    Eigen::Vector3d f_curr_prime = px2cam(pt_curr + epipolar_direction);
    f_curr_prime.normalize();
    double beta_prime = acos(f_curr_prime.dot(-t) / t_norm);
    double gamma = M_PI - alpha - beta_prime;
    double p_prime = t_norm * sin(beta_prime) / sin(gamma);
    double d_cov = p_prime - depth_estimation;
    double d_cov2 = d_cov * d_cov;

    double mu = depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];
    double sigma2 = depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];

    //Previous depth and covariance and New depth (estimated from new image) and covariance

    double mu_fuse = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
    double sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2);

    depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = mu_fuse;
    depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = sigma_fuse2;

    return true;
}

//Plots the crap
void plotDepth(const cv::Mat &depth_truth, const cv::Mat &depth_estimate) {
    cv::imshow("depth_truth", depth_truth * 0.4);
    cv::imshow("depth_estimate", depth_estimate * 0.4);
    // imshow("depth_error", depth_truth - depth_estimate);
    cv::waitKey(1);
}

//evaluates using GT
void evaludateDepth(const cv::Mat &depth_truth, const cv::Mat &depth_estimate) {
    double ave_depth_error = 0;     
    double ave_depth_error_sq = 0;      
    int cnt_depth_data = 0;
    for (int y = border; y < depth_truth.rows - border; y++)
        for (int x = border; x < depth_truth.cols - border; x++) {
            double error = depth_truth.ptr<double>(y)[x] - depth_estimate.ptr<double>(y)[x];
            ave_depth_error += error;
            ave_depth_error_sq += error * error;
            cnt_depth_data++;
        }
    ave_depth_error /= cnt_depth_data;
    ave_depth_error_sq /= cnt_depth_data;

    cout << "Average squared error = " << ave_depth_error_sq << ", average error: " << ave_depth_error << endl;
}

void showEpipolarMatch(const cv::Mat &ref, const cv::Mat &curr, const Eigen::Vector2d &px_ref, const Eigen::Vector2d &px_curr) {
    cout << "Showing epipolar match" ;
    cv::Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, cv::COLOR_GRAY2BGR);
    cv::cvtColor(curr, curr_show, cv::COLOR_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 0, 250), 2);
    cv::circle(curr_show, cv::Point2f(px_curr(0, 0), px_curr(1, 0)), 5, cv::Scalar(0, 0, 250), 2);

    cv::imshow("ref", ref_show);
    cv::imshow("curr", curr_show);
    cv::waitKey(1);
}

void showEpipolarLine(const cv::Mat &ref, const cv::Mat &curr, const Eigen::Vector2d &px_ref, const Eigen::Vector2d &px_min_curr,
                      const Eigen::Vector2d &px_max_curr) {
                        return;

    cv::Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, cv::COLOR_GRAY2BGR);
    cv::cvtColor(curr, curr_show, cv::COLOR_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::line(curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)),
             cv::Scalar(0, 255, 0), 1);

    cv::imshow("ref", ref_show);
    cv::imshow("curr", curr_show);
    cv::waitKey(1);
}
