#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "builtin_interfaces/msg/duration.hpp"
#include "std_msgs/msg/int32.hpp"
#include "std_msgs/msg/float32.hpp"

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl_conversions/pcl_conversions.h>

class ObjectDetectorPCL : public rclcpp::Node
{
public:
  ObjectDetectorPCL()
  : Node("object_detector_pcl"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    declare_parameter<std::string>("input_topic", "/nova/cloud_fused");
    declare_parameter<std::string>("marker_topic", "/cpp/detections");
    declare_parameter<std::string>("target_frame", "base_link");

    // Metrics
    declare_parameter<std::string>("metrics_prefix", "/cpp/metrics");
    declare_parameter<bool>("publish_metrics", true);

    // TF
    declare_parameter<double>("tf_timeout_sec", 0.2);

    // Filtering
    declare_parameter<double>("voxel_leaf", 0.05);
    declare_parameter<int64_t>("sor_mean_k", 20);
    declare_parameter<double>("sor_stddev", 1.0);

    // Plane removal
    declare_parameter<double>("plane_dist_thresh", 0.05);
    declare_parameter<int64_t>("plane_max_iter", 150);

    // Clustering
    declare_parameter<double>("cluster_tolerance", 0.35);
    declare_parameter<int64_t>("cluster_min_size", 20);
    declare_parameter<int64_t>("cluster_max_size", 200000);

    // Z filter
    declare_parameter<double>("min_z", -0.5);
    declare_parameter<double>("max_z", 3.0);

    // Marker
    declare_parameter<double>("marker_lifetime_sec", 0.5);

    input_topic_ = get_parameter("input_topic").as_string();
    marker_topic_ = get_parameter("marker_topic").as_string();
    target_frame_ = get_parameter("target_frame").as_string();

    metrics_prefix_ = get_parameter("metrics_prefix").as_string();
    metrics_enabled_ = get_parameter("publish_metrics").as_bool();

    tf_timeout_sec_ = std::max(0.0, get_parameter("tf_timeout_sec").as_double());

    voxel_leaf_ = get_parameter("voxel_leaf").as_double();
    sor_mean_k_ = clamp_int_(get_parameter("sor_mean_k").as_int(), 1, 1000);
    sor_stddev_ = std::max(0.1, get_parameter("sor_stddev").as_double());

    plane_dist_thresh_ = std::max(0.001, get_parameter("plane_dist_thresh").as_double());
    plane_max_iter_ = clamp_int_(get_parameter("plane_max_iter").as_int(), 10, 5000);

    cluster_tolerance_ = std::max(0.05, get_parameter("cluster_tolerance").as_double());
    cluster_min_size_ = clamp_int_(get_parameter("cluster_min_size").as_int(), 1, 1000000);
    cluster_max_size_ = clamp_int_(get_parameter("cluster_max_size").as_int(), cluster_min_size_, 5000000);

    min_z_ = get_parameter("min_z").as_double();
    max_z_ = get_parameter("max_z").as_double();
    if (max_z_ < min_z_) std::swap(min_z_, max_z_);

    marker_lifetime_sec_ = std::max(0.0, get_parameter("marker_lifetime_sec").as_double());

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, rclcpp::SensorDataQoS(),
      std::bind(&ObjectDetectorPCL::on_cloud, this, std::placeholders::_1));

    pub_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic_, 10);

    if (metrics_enabled_) {
      pub_count_ = create_publisher<std_msgs::msg::Int32>(metrics_prefix_ + "/detections_count", 10);
      pub_ms_ = create_publisher<std_msgs::msg::Float32>(metrics_prefix_ + "/processing_ms", 10);
      pub_points_ = create_publisher<std_msgs::msg::Int32>(metrics_prefix_ + "/points_in", 10);
    }

    RCLCPP_INFO(get_logger(),
      "object_detector_pcl input=%s markers=%s target_frame=%s metrics=%s prefix=%s",
      input_topic_.c_str(), marker_topic_.c_str(), target_frame_.c_str(),
      metrics_enabled_ ? "on" : "off", metrics_prefix_.c_str());
  }

private:
  static int clamp_int_(int64_t v, int lo, int hi)
  {
    if (v < static_cast<int64_t>(lo)) return lo;
    if (v > static_cast<int64_t>(hi)) return hi;
    return static_cast<int>(v);
  }

  static builtin_interfaces::msg::Duration make_duration_(double seconds)
  {
    builtin_interfaces::msg::Duration d;
    if (seconds <= 0.0) { d.sec = 0; d.nanosec = 0; return d; }
    const double s = std::min(seconds, 3600.0);
    const int64_t sec = static_cast<int64_t>(std::floor(s));
    const double frac = s - static_cast<double>(sec);
    const int64_t nsec = static_cast<int64_t>(std::llround(frac * 1e9));
    d.sec = static_cast<int32_t>(sec);
    d.nanosec = static_cast<uint32_t>(std::max<int64_t>(0, std::min<int64_t>(nsec, 999999999)));
    return d;
  }

  static float elapsed_ms_(const std::chrono::steady_clock::time_point & t0)
  {
    const auto t1 = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> dt = t1 - t0;
    return static_cast<float>(dt.count());
  }

  bool transform_cloud_to_target_(
    const sensor_msgs::msg::PointCloud2 & in,
    sensor_msgs::msg::PointCloud2 & out)
  {
    if (in.header.frame_id.empty()) return false;

    if (in.header.frame_id == target_frame_) {
      out = in;
      return true;
    }

    try {
      const rclcpp::Time stamp(in.header.stamp);
      const auto tf = tf_buffer_.lookupTransform(
        target_frame_, in.header.frame_id, stamp,
        rclcpp::Duration::from_seconds(tf_timeout_sec_));
      tf2::doTransform(in, out, tf);
      out.header.frame_id = target_frame_;
      return true;
    } catch (const std::exception & e) {
      try {
        const auto tf = tf_buffer_.lookupTransform(target_frame_, in.header.frame_id, tf2::TimePointZero);
        tf2::doTransform(in, out, tf);
        out.header.frame_id = target_frame_;
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
          "TF at stamp failed (%s). Used latest TF instead.", e.what());
        return true;
      } catch (const std::exception & e2) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
          "TF failed: %s (and latest failed: %s). Skipping cloud.", e.what(), e2.what());
        return false;
      }
    }
  }

  void publish_metrics_msgs_(int points_in, int detections, float ms)
  {
    if (!metrics_enabled_) return;

    std_msgs::msg::Int32 c; c.data = detections;
    std_msgs::msg::Float32 t; t.data = ms;
    std_msgs::msg::Int32 p; p.data = points_in;

    pub_count_->publish(c);
    pub_ms_->publish(t);
    pub_points_->publish(p);
  }

  void publish_empty_markers_()
  {
    visualization_msgs::msg::MarkerArray out;
    pub_markers_->publish(out);
  }

  void on_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    const auto t0 = std::chrono::steady_clock::now();

    sensor_msgs::msg::PointCloud2 cloud_in_target;
    if (!transform_cloud_to_target_(*msg, cloud_in_target)) {
      publish_empty_markers_();
      publish_metrics_msgs_(0, 0, 0.0f);
      return;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZI>());
    try {
      pcl::fromROSMsg(cloud_in_target, *cloud_in);
    } catch (...) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "Failed to convert PointCloud2 to PCL cloud.");
      publish_empty_markers_();
      publish_metrics_msgs_(0, 0, 0.0f);
      return;
    }

    const int points_in = static_cast<int>(cloud_in->size());
    if (!cloud_in || cloud_in->empty()) {
      publish_empty_markers_();
      publish_metrics_msgs_(points_in, 0, elapsed_ms_(t0));
      return;
    }

    // Z filter
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_z(new pcl::PointCloud<pcl::PointXYZI>());
    cloud_z->points.reserve(cloud_in->points.size());
    for (const auto & p : cloud_in->points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
      if (p.z < min_z_ || p.z > max_z_) continue;
      cloud_z->points.push_back(p);
    }
    cloud_z->width = static_cast<uint32_t>(cloud_z->points.size());
    cloud_z->height = 1;
    cloud_z->is_dense = true;

    if (cloud_z->empty()) {
      publish_empty_markers_();
      publish_metrics_msgs_(points_in, 0, elapsed_ms_(t0));
      return;
    }

    // Voxel downsample
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ds(new pcl::PointCloud<pcl::PointXYZI>());
    if (voxel_leaf_ > 1e-6) {
      pcl::VoxelGrid<pcl::PointXYZI> vg;
      vg.setInputCloud(cloud_z);
      vg.setLeafSize(static_cast<float>(voxel_leaf_), static_cast<float>(voxel_leaf_), static_cast<float>(voxel_leaf_));
      vg.filter(*cloud_ds);
    } else {
      cloud_ds = cloud_z;
    }

    if (!cloud_ds || cloud_ds->empty()) {
      publish_empty_markers_();
      publish_metrics_msgs_(points_in, 0, elapsed_ms_(t0));
      return;
    }

    // Outlier removal
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZI>());
    {
      pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
      sor.setInputCloud(cloud_ds);
      sor.setMeanK(sor_mean_k_);
      sor.setStddevMulThresh(sor_stddev_);
      sor.filter(*cloud_f);
    }

    if (!cloud_f || cloud_f->empty()) {
      publish_empty_markers_();
      publish_metrics_msgs_(points_in, 0, elapsed_ms_(t0));
      return;
    }

    // Plane removal
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_nonground(new pcl::PointCloud<pcl::PointXYZI>());
    {
      pcl::SACSegmentation<pcl::PointXYZI> seg;
      seg.setOptimizeCoefficients(true);
      seg.setModelType(pcl::SACMODEL_PLANE);
      seg.setMethodType(pcl::SAC_RANSAC);
      seg.setMaxIterations(plane_max_iter_);
      seg.setDistanceThreshold(plane_dist_thresh_);

      pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
      pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients());

      seg.setInputCloud(cloud_f);
      seg.segment(*inliers, *coeff);

      if (!inliers || inliers->indices.empty()) {
        cloud_nonground = cloud_f;
      } else {
        pcl::ExtractIndices<pcl::PointXYZI> extract;
        extract.setInputCloud(cloud_f);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*cloud_nonground);
      }
    }

    if (!cloud_nonground || cloud_nonground->empty()) {
      publish_empty_markers_();
      publish_metrics_msgs_(points_in, 0, elapsed_ms_(t0));
      return;
    }

    // Clustering
    std::vector<pcl::PointIndices> cluster_indices;
    {
      pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>());
      tree->setInputCloud(cloud_nonground);

      pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
      ec.setClusterTolerance(cluster_tolerance_);
      ec.setMinClusterSize(cluster_min_size_);
      ec.setMaxClusterSize(cluster_max_size_);
      ec.setSearchMethod(tree);
      ec.setInputCloud(cloud_nonground);
      ec.extract(cluster_indices);
    }

    visualization_msgs::msg::MarkerArray out;
    out.markers.reserve(cluster_indices.size());

    const auto stamp = this->now();
    int id = 0;

    for (const auto & idxs : cluster_indices) {
      if (idxs.indices.empty()) continue;

      float minx = std::numeric_limits<float>::infinity();
      float miny = std::numeric_limits<float>::infinity();
      float minz = std::numeric_limits<float>::infinity();
      float maxx = -std::numeric_limits<float>::infinity();
      float maxy = -std::numeric_limits<float>::infinity();
      float maxz = -std::numeric_limits<float>::infinity();

      for (int i : idxs.indices) {
        const auto & p = cloud_nonground->points[static_cast<size_t>(i)];
        minx = std::min(minx, p.x); miny = std::min(miny, p.y); minz = std::min(minz, p.z);
        maxx = std::max(maxx, p.x); maxy = std::max(maxy, p.y); maxz = std::max(maxz, p.z);
      }

      const float sx = maxx - minx;
      const float sy = maxy - miny;
      const float sz = maxz - minz;

      if (!(std::isfinite(sx) && std::isfinite(sy) && std::isfinite(sz))) continue;
      if (sx < 0.02f || sy < 0.02f || sz < 0.02f) continue;

      visualization_msgs::msg::Marker m;
      m.header = cloud_in_target.header;
      m.header.stamp = stamp;
      m.ns = "pcl_detections";
      m.id = id++;
      m.type = visualization_msgs::msg::Marker::CUBE;
      m.action = visualization_msgs::msg::Marker::ADD;

      m.pose.position.x = (minx + maxx) * 0.5f;
      m.pose.position.y = (miny + maxy) * 0.5f;
      m.pose.position.z = (minz + maxz) * 0.5f;
      m.pose.orientation.w = 1.0;

      m.scale.x = std::max(0.05f, sx);
      m.scale.y = std::max(0.05f, sy);
      m.scale.z = std::max(0.05f, sz);

      m.color.r = 1.0f;
      m.color.g = 0.5f;
      m.color.b = 0.1f;
      m.color.a = 0.6f;

      if (marker_lifetime_sec_ > 0.0) m.lifetime = make_duration_(marker_lifetime_sec_);

      out.markers.push_back(m);
    }

    pub_markers_->publish(out);

    const float ms = elapsed_ms_(t0);
    publish_metrics_msgs_(points_in, static_cast<int>(out.markers.size()), ms);

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
      "PCL detector (%s): points=%d clusters=%zu time=%.2fms",
      input_topic_.c_str(), points_in, out.markers.size(), ms);
  }

  std::string input_topic_;
  std::string marker_topic_;
  std::string target_frame_;

  std::string metrics_prefix_;
  bool metrics_enabled_;

  double tf_timeout_sec_;

  double voxel_leaf_;
  int sor_mean_k_;
  double sor_stddev_;

  double plane_dist_thresh_;
  int plane_max_iter_;

  double cluster_tolerance_;
  int cluster_min_size_;
  int cluster_max_size_;

  double min_z_;
  double max_z_;
  double marker_lifetime_sec_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;

  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr pub_count_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_ms_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr pub_points_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ObjectDetectorPCL>());
  rclcpp::shutdown();
  return 0;
}
