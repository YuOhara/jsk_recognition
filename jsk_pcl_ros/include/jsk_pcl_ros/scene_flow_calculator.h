// -*- mode: c++ -*-
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
 *     contributors may be used to endorse or promote products derived
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


#ifndef JSK_PCL_SCENE_FLOW_CALCULATOR_H_
#define JSK_PCL_SCENE_FLOW_CALCULATOR_H_

#include <pcl_ros/pcl_nodelet.h>
#include <dynamic_reconfigure/server.h>
#include <jsk_recognition_msgs/BoundingBox.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <jsk_topic_tools/connection_based_nodelet.h>
#include <jsk_recognition_msgs/PointsArray.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <jsk_recognition_utils/time_util.h>
#include <jsk_pcl_ros/PD-Flow/pdflow_cudalib.h>
#include <image_geometry/pinhole_camera_model.h>
#include "jsk_pcl_ros/tf_listener_singleton.h"
#include <cv_bridge/cv_bridge.h>

using Eigen::MatrixXf;

namespace jsk_pcl_ros
{
  class SceneFlowCalculator: public jsk_topic_tools::ConnectionBasedNodelet
  {
  public:
    typedef pcl::PointXYZRGBNormal PointT;
    typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image,
      sensor_msgs::Image > SyncPolicy;
    SceneFlowCalculator(): timer_(10), done_init_(false), done_sub_caminfo_(false), calc_phase_(false) { }
    void subImages(
      const sensor_msgs::Image::ConstPtr& image_msg,
      const sensor_msgs::Image::ConstPtr& depth_msg);
    virtual void cameraInfoCallback(
      const sensor_msgs::CameraInfo::ConstPtr& msg);
    tf::TransformListener* tf_listener_;
  protected:
    virtual void onInit();
    virtual void subscribe();
    virtual void unsubscribe();
    ros::Subscriber sub_camera_info_;
    ros::Publisher pub_result_cloud_;
    message_filters::Subscriber<sensor_msgs::Image> sub_input_;
    message_filters::Subscriber<sensor_msgs::Image> sub_box_;
    jsk_recognition_utils::WallDurationTimer timer_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;
    image_geometry::PinholeCameraModel model_;
    std_msgs::Header header_;
    std::string base_frame_id_;
    Eigen::Affine3f trans_from_base_old_;
    Eigen::Affine3f trans_from_base_now_;
    bool done_init_;
    bool done_sub_caminfo_;
    bool calc_phase_;
    boost::mutex mutex_;
    float g_mask_[25];
    float mu_, lambda_i_, lambda_d_;
    unsigned int ctf_levels_; //Number of levels used in the coarse-to-fine scheme (always dividing by two)
    unsigned int num_max_iter_[6]; //Max number of iterations distributed homogeneously between all levels
    MatrixXf colour_wf_;
    MatrixXf depth_wf_;
    
    //Matrices that store the images downsampled
    std::vector<MatrixXf> colour_;
    std::vector<MatrixXf> colour_old_;
    std::vector<MatrixXf> depth_;
    std::vector<MatrixXf> depth_old_;
    std::vector<MatrixXf> xx_;
    std::vector<MatrixXf> xx_old_;
    std::vector<MatrixXf> yy_;
    std::vector<MatrixXf> yy_old_;


    //Motion field
    std::vector<MatrixXf> dx_;
    std::vector<MatrixXf> dy_;
    std::vector<MatrixXf> dz_;

    unsigned int rows_, cols_, width_, height_;
    float fovh_, fovv_;

    cv::Mat image_float_, colour_float_, depth_float_;

    //Cuda
    CSF_cuda csf_host_, *csf_device_;
    void solveSceneFlowGPU();
    void capture(
                 const sensor_msgs::Image::ConstPtr& image_msg,
                 const sensor_msgs::Image::ConstPtr& depth_msg);
    void createImagePyramidGPU();
    void initializeCUDA();
    void publishScene();
  private:

  };
}

#endif
