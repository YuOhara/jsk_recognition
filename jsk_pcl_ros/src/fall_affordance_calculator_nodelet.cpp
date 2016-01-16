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


#include "jsk_pcl_ros/fall_affordance_calculator.h"
#include "jsk_recognition_utils/pcl_conversion_util.h"
#include <pcl/filters/passthrough.h>
#include <pcl/common/centroid.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl/common/transforms.h>
#include <eigen_conversions/eigen_msg.h>
#include "jsk_pcl_ros_utils/transform_pointcloud_in_bounding_box.h"
#include <jsk_recognition_utils/pcl_ros_util.h>

namespace jsk_pcl_ros
{
  void FallAffordanceCalculator::onInit()
  {
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    ConnectionBasedNodelet::onInit();
    tf_listener_ = TfListenerSingleton::getInstance();
    ////////////////////////////////////////////////////////
    // Publishers
    ////////////////////////////////////////////////////////
    srv_get_fall_affordance_ = pnh_->advertiseService(
      "get_fall_affordance", &FallAffordanceCalculator::alignWithBoxService, this);
    onInitPostProcess();
    ////////////////////////////////////////////////////////
    // Subscription
    ////////////////////////////////////////////////////////
    sub_input_.subscribe(*pnh_, "input", 1);
    sub_box_.subscribe(*pnh_, "input_box", 1);
    sync_ = boost::make_shared<message_filters::Synchronizer<SyncPolicy> >(100);
    sync_->connectInput(sub_input_, sub_box_);
    sync_->registerCallback(boost::bind(
                                        &FallAffordanceCalculator::alignWithBox,
                                        this, _1, _2));
  }

  void FallAffordanceCalculator::subscribe()
  {
  }

  void FallAffordanceCalculator::unsubscribe()
  {
  }

  void FallAffordanceCalculator::alignWithBox(
      const sensor_msgs::PointCloud2::ConstPtr& msg,
      const jsk_recognition_msgs::BoundingBox::ConstPtr& box_msg)
  {
    boost::mutex::scoped_lock lock(mutex_);
    try
    {
      pcl::PointCloud<PointT>::Ptr input (new pcl::PointCloud<PointT>);
      jsk_pcl_ros_utils::transformPointcloudInBoundingBox<PointT>(
        *box_msg, *msg,
        *input, offset_,
        *tf_listener_);
      reference_cloud_ = input;
      Eigen::Affine3f inversed_offset_ = offset_.inverse();
      frame_id_ = box_msg->header.frame_id;
    }
    catch (tf2::ConnectivityException &e)
    {
      JSK_NODELET_ERROR("[%s] Transform error: %s", __PRETTY_FUNCTION__, e.what());
      return;
    }
    catch (tf2::InvalidArgumentException &e)
    {
      JSK_NODELET_ERROR("[%s] Transform error: %s", __PRETTY_FUNCTION__, e.what());
      return;
    }
    box_height_ = box_msg->dimensions.z;
    pcl::PassThrough<PointT> pass;
    pass.setInputCloud(reference_cloud_);
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*reference_cloud_, centroid);
    centroid_z_ = centroid[2];
    pcl::PointCloud<PointT>::Ptr cut_cloud(new pcl::PointCloud<PointT>);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-box_height_, -box_height_ + 0.01);
    pass.filter(*cut_cloud);
    cut_cloud_ = cut_cloud;
  }
  
  bool FallAffordanceCalculator::alignWithBoxService(
    jsk_pcl_ros::GetFallAffordance::Request& req,
    jsk_pcl_ros::GetFallAffordance::Response& res)
  {
    // assume that hand moves horizontally
    // not consider of collision appearance point
    boost::mutex::scoped_lock lock(mutex_);
    ros::Time now = ros::Time::now();
    geometry_msgs::PoseStamped trans_pose;
    try{
      tf_listener_->waitForTransform(frame_id_, req.grasp_pose_stamped.header.frame_id, now, ros::Duration(2.0));
      tf_listener_->transformPose(frame_id_, now, req.grasp_pose_stamped, req.grasp_pose_stamped.header.frame_id, trans_pose);
    }
    catch (tf::TransformException ex){
      JSK_NODELET_ERROR("TF Error %s",ex.what());
      return false;
    }
    Eigen::Affine3f trans_pose_eigened;
    tf::poseMsgToEigen(trans_pose.pose,
                       trans_pose_eigened);
    Eigen::Affine3f pose_eigened = offset_ * trans_pose_eigened;
    pcl::PointCloud<PointT>::Ptr output_cloud(new pcl::PointCloud<PointT>);
    pcl::transformPointCloud(*cut_cloud_, *output_cloud, pose_eigened);
    float max_x = -200, min_x = 200;
    for(size_t index=0; index<output_cloud->size(); index++){
      if (max_x < output_cloud->points[index].x)
        max_x = output_cloud->points[index].x;
      if (min_x > output_cloud->points[index].x)
        min_x = output_cloud->points[index].x;
    }
    JSK_NODELET_INFO("min_x %f max_x %f", min_x, max_x);
    float grasp_z = pose_eigened.translation()[2];
    JSK_NODELET_INFO("grasp_height: %f", grasp_z);
    res.affordable_distance = (max_x - min_x) * grasp_z / centroid_z_ / 2;
    JSK_NODELET_INFO("l=%f, hand_z=%f, centroid_z=%f, affordance=%f", max_x - min_x, grasp_z, centroid_z_, res.affordable_distance);
    return true;
  }
}

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS (jsk_pcl_ros::FallAffordanceCalculator, nodelet::Nodelet);
