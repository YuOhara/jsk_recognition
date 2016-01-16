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


#ifndef JSK_PCL_ROS_FALL_AFFORDANCE_CALCULATOR_H_
#define JSK_PCL_ROS_FALL_AFFORDANCE_CALCULATOR_H_

#include <pcl_ros/pcl_nodelet.h>
#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_pcl_ros/GetFallAffordance.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include "jsk_pcl_ros/tf_listener_singleton.h"
#include <jsk_topic_tools/connection_based_nodelet.h>
#include <jsk_recognition_msgs/PointsArray.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Float32.h>

namespace jsk_pcl_ros
{
  class FallAffordanceCalculator: public jsk_topic_tools::ConnectionBasedNodelet
  {
  public:
    typedef pcl::PointXYZRGBNormal PointT;
    typedef message_filters::sync_policies::ExactTime<
      sensor_msgs::PointCloud2,
      jsk_recognition_msgs::BoundingBox > SyncPolicy;
    FallAffordanceCalculator() { }
  protected:
    ////////////////////////////////////////////////////////
    // methosd
    ////////////////////////////////////////////////////////
    virtual void onInit();
    virtual void registerPoint(
      const sensor_msgs::PointCloud2::ConstPtr& msg,
      const jsk_recognition_msgs::BoundingBox::ConstPtr& box_msg);
    virtual bool getFallAffordanceService(
      jsk_pcl_ros::GetFallAffordance::Request& req,
      jsk_pcl_ros::GetFallAffordance::Response& res);
    virtual void subscribe();
    virtual void unsubscribe();
    ////////////////////////////////////////////////////////
    // ROS variables
    ////////////////////////////////////////////////////////
    ros::ServiceServer srv_get_fall_affordance_;
    boost::mutex mutex_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_input_;
    message_filters::Subscriber<jsk_recognition_msgs::BoundingBox> sub_box_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;
    tf::TransformListener* tf_listener_;
    pcl::PointCloud<PointT>::Ptr reference_cloud_;
    pcl::PointCloud<PointT>::Ptr cut_cloud_;
    Eigen::Affine3f offset_;
    Eigen::Affine3f inversed_offset_;
    float box_height_;
    std::string frame_id_;
    float centroid_z_;
  private:
  };
}

#endif
