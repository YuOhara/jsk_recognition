// -*- mode: c++; indent-tabs-mode: nil; -*-
/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2016, JSK Lab
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/o2r other materials provided
 *     with the distribution.
 *   * Neither the name of the JSK Lab nor the names of its
 *     contributors may be used to eandorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

#include "jsk_pcl_ros/scene_flow_calculator.h"
#include "jsk_recognition_utils/pcl_conversion_util.h"
#include <eigen_conversions/eigen_msg.h>
#include <pcl/common/transforms.h>
#include <eigen_conversions/eigen_msg.h>
#include "jsk_pcl_ros_utils/transform_pointcloud_in_bounding_box.h"
#include <image_geometry/pinhole_camera_model.h>
#include <pcl/registration/correspondence_estimation_organized_projection.h>
#include <pcl/registration/correspondence_estimation_normal_shooting.h>
#include <jsk_recognition_utils/pcl_ros_util.h>
#include <math.h>
namespace jsk_pcl_ros
{
  void SceneFlowCalculator::onInit()
  {
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    ConnectionBasedNodelet::onInit();
    done_init_ = true;
    sub_camera_info_ = pnh_->subscribe("info", 1,
                                  &SceneFlowCalculator::cameraInfoCallback,
                                  this);

    pub_result_cloud_ = advertise<sensor_msgs::PointCloud2>(*pnh_,
      "output", 1);
    onInitPostProcess();
  }
  
  void SceneFlowCalculator::subscribe()
  {
    ////////////////////////////////////////////////////////
    // Subscription
    ////////////////////////////////////////////////////////
    sub_input_.subscribe(*pnh_, "image", 1);
    sub_box_.subscribe(*pnh_, "depth", 1);
    sync_ = boost::make_shared<message_filters::Synchronizer<SyncPolicy> >(100);
    sync_->connectInput(sub_input_, sub_box_);
    sync_->registerCallback(boost::bind(
                                        &SceneFlowCalculator::subImages,
                                        this, _1, _2));
  }

  void SceneFlowCalculator::unsubscribe()
  {
    sub_input_.unsubscribe();
    sub_box_.unsubscribe();
  }

  void SceneFlowCalculator::cameraInfoCallback(
    const sensor_msgs::CameraInfo::ConstPtr& msg)
  {
    boost::mutex::scoped_lock lock(mutex_);
    model_.fromCameraInfo(msg);
    rows_ = msg->height;
    cols_ = msg->width;
    fovh_ = 2 * atan(model_.cx()/model_.fx());
    fovv_ = 2 * atan(model_.cy()/model_.fy());
    ctf_levels_ = round(log2(rows_/15)) + 1;
    //Maximum value set to 100 at the finest level
    for (int i=5; i>=0; i--)
    {
      if (i >= ctf_levels_ - 1)
        num_max_iter_[i] = 100;	
      else
        num_max_iter_[i] = num_max_iter_[i+1]-15;
    }
    float v_mask[5] = {1.f,4.f,6.f,4.f,1.f};
    for (unsigned int i=0; i<5; i++)
      for (unsigned int j=0; j<5; j++)
        g_mask_[i+5*j] = v_mask[i]*v_mask[j]/256.f;

    colour_wf_ = MatrixXf::Zero(rows_, cols_);
    depth_wf_ = MatrixXf::Zero(rows_, cols_);
    //Resize vectors according to levels
    dx_.resize(ctf_levels_); dy_.resize(ctf_levels_); dz_.resize(ctf_levels_);

    const unsigned int width = colour_wf_.cols();
    const unsigned int height = colour_wf_.rows();
    unsigned int s, cols_i, rows_i;

    for (unsigned int i = 0; i<ctf_levels_; i++)
    {
        s = pow(2.f,int(ctf_levels_-(i+1)));
        cols_i = cols_/s; rows_i = rows_/s;
        dx_[ctf_levels_-i-1] = MatrixXf::Zero(rows_i,cols_i);
        dy_[ctf_levels_-i-1] = MatrixXf::Zero(rows_i,cols_i);
        dz_[ctf_levels_-i-1] = MatrixXf::Zero(rows_i,cols_i);
    }
    //Resize pyramid
    const unsigned int pyr_levels = round(log2(width/cols_)) + ctf_levels_;
    colour_.resize(pyr_levels);
    colour_old_.resize(pyr_levels);
    depth_.resize(pyr_levels);
    depth_old_.resize(pyr_levels);
    xx_.resize(pyr_levels);
    xx_old_.resize(pyr_levels);
    yy_.resize(pyr_levels);
    yy_old_.resize(pyr_levels);
    for (unsigned int i = 0; i<pyr_levels; i++)
    {
      s = pow(2.f,int(i));
      colour_[i].resize(height/s, width/s);
      colour_old_[i].resize(height/s, width/s);
      colour_[i].derived().setConstant(0.0f);
      colour_old_[i].derived().setConstant(0.0f);
      depth_[i].resize(height/s, width/s);
      depth_old_[i].resize(height/s, width/s);
      depth_[i].derived().setConstant(0.0f);
      depth_old_[i].derived().setConstant(0.0f);
      xx_[i].resize(height/s, width/s);
      xx_old_[i].resize(height/s, width/s);
      xx_[i].derived().setConstant(0.0f);
      xx_old_[i].derived().setConstant(0.0f);
      yy_[i].resize(height/s, width/s);
      yy_old_[i].resize(height/s, width/s);
      yy_[i].derived().setConstant(0.0f);
      yy_old_[i].derived().setConstant(0.0f);
    }
    //Parameters of the variational method
    lambda_i_ = 0.04f;
    lambda_d_ = 0.35f;
    mu_ = 75.f;
    initializeCUDA();

    sub_camera_info_.shutdown();
    done_sub_caminfo_ = true;
  }
  void SceneFlowCalculator::initializeCUDA(){
    //Read parameters
    csf_host_.readParameters(rows_, cols_, lambda_i_, lambda_d_, mu_, g_mask_, ctf_levels_, (unsigned int) 1, fovh_, fovv_);
    //Allocate memory
    csf_host_.allocateDevMemory();

  }
  void SceneFlowCalculator::subImages(
      const sensor_msgs::Image::ConstPtr& image_msg,
      const sensor_msgs::Image::ConstPtr& depth_msg)
  {
    boost::mutex::scoped_lock lock(mutex_);
    if (!done_init_) {
      JSK_NODELET_WARN("not yet initialized");
      return;
    }
    if (!done_sub_caminfo_) {
      JSK_NODELET_WARN("not yet sub caminfo");
      return;
    }
    capture(image_msg, depth_msg);
    createImagePyramidGPU();
    if (calc_phase_) {
      solveSceneFlowGPU();
      publishScene();
    }
    // update scene
    calc_phase_ = true;

  }
  void SceneFlowCalculator::capture(
               const sensor_msgs::Image::ConstPtr& image_msg,
               const sensor_msgs::Image::ConstPtr& depth_msg){
    cv_bridge::CvImagePtr cv_image_ptr;
    cv_bridge::CvImagePtr cv_depth_ptr;
    try{
      cv_image_ptr = cv_bridge::toCvCopy(image_msg, "bgr8");
      cv_depth_ptr = cv_bridge::toCvCopy(depth_msg, "32FC1");
    }
    catch (cv_bridge::Exception& e){
      JSK_NODELET_ERROR("error in converting msg->cv_mat: %s", e.what());
    }
    header_ = depth_msg->header;
    //cv_depth_ptr->image.convertTo(depth_float, CV_32FC1);
    depth_float_ = cv_depth_ptr->image.clone();
    image_float_ = cv_image_ptr->image.clone();
    cvtColor(image_float_, colour_float_ ,CV_RGB2GRAY);
    for (unsigned int v=0; v<colour_wf_.cols(); v++)
      for (unsigned int u=0; u<colour_wf_.rows(); u++){
        depth_wf_(u, v) = depth_float_.at<float>(u, v) * 1;
        colour_wf_(u, v) = (float) colour_float_.at<unsigned char>(u, v);
      }
  }
  void SceneFlowCalculator::createImagePyramidGPU(){
    //Copy new frames to the scene flow object
    csf_host_.copyNewFrames(colour_wf_.data(), depth_wf_.data());
    //Copy scene flow object to device
    csf_device_ = ObjectToDevice(&csf_host_);
    unsigned int pyr_levels = round(log2(640/(1*cols_))) + ctf_levels_;
    GaussianPyramidBridge(csf_device_, pyr_levels, 1);
    //Copy scene flow object back to host
    BridgeBack(&csf_host_, csf_device_);
  }

  void SceneFlowCalculator::solveSceneFlowGPU()
  {
    //Define variables
    unsigned int s;
    unsigned int cols_i, rows_i;
    unsigned int level_image;
    unsigned int num_iter;

    //For every level (coarse-to-fine)
    for (unsigned int i=0; i<ctf_levels_; i++)
    {
      const unsigned int width = colour_wf_.cols();
      s = pow(2.f,int(ctf_levels_-(i+1)));
      cols_i = cols_/s;
      rows_i = rows_/s;
      level_image = ctf_levels_ - i + round(log2(width/cols_)) - 1;
      //=========================================================================
      //                              Cuda - Begin
      //=========================================================================

      //Cuda allocate memory
      csf_host_.allocateMemoryNewLevel(rows_i, cols_i, i, level_image);
      //Cuda copy object to device
      csf_device_ = ObjectToDevice(&csf_host_);
      //Assign zeros to the corresponding variables
      AssignZerosBridge(csf_device_);
      //Upsample previous solution
      if (i>0)
        UpsampleBridge(csf_device_);
        //Compute connectivity (Rij)
      RijBridge(csf_device_);
      //Compute colour and depth derivatives
      ImageGradientsBridge(csf_device_);
      WarpingBridge(csf_device_);
      //Compute mu_uv and step sizes for the primal-dual algorithm
      MuAndStepSizesBridge(csf_device_);
      //Primal-Dual solver
      for (num_iter = 0; num_iter < num_max_iter_[i]; num_iter++)
      {
        GradientBridge(csf_device_);
        DualVariablesBridge(csf_device_);
        DivergenceBridge(csf_device_);
        PrimalVariablesBridge(csf_device_);
      }
      //Filter solution
      FilterBridge(csf_device_);
      //Compute the motion field
      MotionFieldBridge(csf_device_);
      //BridgeBack
      BridgeBack(&csf_host_, csf_device_);
      //Free variables of variables associated to this level
      csf_host_.freeLevelVariables();
      //Copy motion field and images to CPU
      csf_host_.copyAllSolutions(dx_[ctf_levels_-i-1].data(), dy_[ctf_levels_-i-1].data(), dz_[ctf_levels_-i-1].data(),
                                depth_[level_image].data(), depth_old_[level_image].data(), colour_[level_image].data(), colour_old_[level_image].data(),
                                xx_[level_image].data(), xx_old_[level_image].data(), yy_[level_image].data(), yy_old_[level_image].data());
      //For debugging
      //DebugBridge(csf_device);
      //=========================================================================
      //                              Cuda - end
      //=========================================================================
    }
  }
  void SceneFlowCalculator::publishScene(){
    float center_x = model_.cx();
    float center_y = model_.cy();

    float unit_scaling = 1.0;
    float constant_x = unit_scaling / model_.fx();
    float constant_y = unit_scaling / model_.fy();

    float bad_point = std::numeric_limits<float>::quiet_NaN ();
    pcl::PointCloud<pcl::PointXYZRGBNormal> cloud;
    unsigned int width = depth_float_.cols;
    unsigned int height = depth_float_.rows;
    cloud.points.resize(width * height);
    cloud.width = width;
    cloud.height = height;
    for (size_t i=0; i < height; i++) {
      for (size_t j=0; j < width; j++) {
        float depth = depth_float_.at<float>(i, j);
        if (! std::isfinite(depth)){
          cloud.points[i * width + j].x = bad_point;
          cloud.points[i * width + j].y = bad_point;
          cloud.points[i * width + j].z = bad_point;
        }
        else{
          cloud.points[i * width + j].x = (j - center_x) * depth * constant_x;
          cloud.points[i * width + j].y = (i - center_y) * depth * constant_y;
          cloud.points[i * width + j].z = depth;
        }
        cloud.points[i * width + j].b = image_float_.at<cv::Vec3b>(i, j)[0];
        cloud.points[i * width + j].g = image_float_.at<cv::Vec3b>(i, j)[1];
        cloud.points[i * width + j].r = image_float_.at<cv::Vec3b>(i, j)[2];
        cloud.points[i * width + j].normal_x = dx_[0](i, j);
        cloud.points[i * width + j].normal_x = dy_[0](i, j);
        cloud.points[i * width + j].normal_x = dz_[0](i, j);
      }
    }
    sensor_msgs::PointCloud2 ros_out;
    pcl::toROSMsg(cloud, ros_out);
    ros_out.header = header_;
    pub_result_cloud_.publish(ros_out);
  }
}

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS (jsk_pcl_ros::SceneFlowCalculator, nodelet::Nodelet);
