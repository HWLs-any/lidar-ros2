#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

class CloudMonitorCpp : public rclcpp::Node
{
public:
  CloudMonitorCpp() : Node("cloud_monitor_cpp")
  {
    declare_parameter<std::string>("cloud_topic", "/front/cloud");
    declare_parameter<double>("log_hz", 1.0);
    declare_parameter<std::string>("expected_frame_id", "");

    cloud_topic_ = get_parameter("cloud_topic").as_string();
    log_hz_ = std::max(0.1, get_parameter("log_hz").as_double());
    expected_frame_id_ = get_parameter("expected_frame_id").as_string();

    const auto sensor_qos = rclcpp::SensorDataQoS();

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic_, sensor_qos,
      std::bind(&CloudMonitorCpp::on_cloud, this, std::placeholders::_1));

    last_log_time_ = now();
    last_msg_count_ = 0;

    RCLCPP_INFO(get_logger(), "cloud_monitor_cpp subscribing to %s (log_hz=%.2f)",
      cloud_topic_.c_str(), log_hz_);
  }

private:
  void on_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    last_msg_count_++;

    if (!expected_frame_id_.empty() && msg->header.frame_id != expected_frame_id_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "frame_id mismatch: got '%s' expected '%s'",
        msg->header.frame_id.c_str(), expected_frame_id_.c_str());
    }

    // Defensive: ensure x,y,z fields exist (PointCloud2Iterator throws otherwise)
    if (!has_field(*msg, "x") || !has_field(*msg, "y") || !has_field(*msg, "z")) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "PointCloud2 missing x/y/z fields. Cannot compute stats.");
      return;
    }

    // Compute basic stats (min/avg/max range). Skip NaNs safely.
    std::size_t n = 0;
    double min_r = std::numeric_limits<double>::infinity();
    double max_r = 0.0;
    double sum_r = 0.0;

    sensor_msgs::PointCloud2ConstIterator<float> ix(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iy(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iz(*msg, "z");

    for (; ix != ix.end(); ++ix, ++iy, ++iz) {
      const float x = *ix;
      const float y = *iy;
      const float z = *iz;

      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        continue;
      }
      const double r = std::sqrt(double(x)*x + double(y)*y + double(z)*z);
      if (!std::isfinite(r)) {
        continue;
      }
      n++;
      sum_r += r;
      min_r = std::min(min_r, r);
      max_r = std::max(max_r, r);
    }

    const auto t = now();
    const double elapsed = (t - last_log_time_).seconds();
    const double period = 1.0 / log_hz_;

    if (elapsed >= period) {
      const double rate = last_msg_count_ / std::max(1e-9, elapsed);
      const double avg_r = (n > 0) ? (sum_r / double(n)) : 0.0;

      RCLCPP_INFO(get_logger(),
        "[%s] rate=%.2f Hz, points=%zu, range min/avg/max=%.2f/%.2f/%.2f m, frame='%s'",
        cloud_topic_.c_str(), rate, n,
        std::isfinite(min_r) ? min_r : 0.0, avg_r, max_r,
        msg->header.frame_id.c_str());

      last_log_time_ = t;
      last_msg_count_ = 0;
    }
  }

  static bool has_field(const sensor_msgs::msg::PointCloud2 & msg, const std::string & name)
  {
    for (const auto & f : msg.fields) {
      if (f.name == name) {
        return true;
      }
    }
    return false;
  }

  std::string cloud_topic_;
  double log_hz_;
  std::string expected_frame_id_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;

  rclcpp::Time last_log_time_;
  std::size_t last_msg_count_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CloudMonitorCpp>());
  rclcpp::shutdown();
  return 0;
}
