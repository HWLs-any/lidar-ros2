#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

struct VoxelKey
{
  int64_t x;
  int64_t y;
  int64_t z;

  bool operator==(const VoxelKey & other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash
{
  std::size_t operator()(const VoxelKey & k) const noexcept
  {
    // Simple combine hash (safe, fast enough for our use)
    const std::size_t h1 = std::hash<int64_t>{}(k.x);
    const std::size_t h2 = std::hash<int64_t>{}(k.y);
    const std::size_t h3 = std::hash<int64_t>{}(k.z);
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};

class CloudFilterCpp : public rclcpp::Node
{
public:
  CloudFilterCpp() : Node("cloud_filter_cpp")
  {
    declare_parameter<std::string>("input_topic", "/front/cloud");
    declare_parameter<std::string>("output_topic", "/front/cloud_filtered");
    declare_parameter<double>("min_range", 0.2);
    declare_parameter<double>("max_range", 80.0);
    declare_parameter<double>("min_z", -std::numeric_limits<double>::infinity());
    declare_parameter<double>("max_z",  std::numeric_limits<double>::infinity());
    declare_parameter<double>("voxel_size", 0.05);  // 0 disables voxel filtering
    declare_parameter<std::string>("output_frame_id", ""); // empty => keep input

    input_topic_ = get_parameter("input_topic").as_string();
    output_topic_ = get_parameter("output_topic").as_string();
    min_range_ = get_parameter("min_range").as_double();
    max_range_ = get_parameter("max_range").as_double();
    min_z_ = get_parameter("min_z").as_double();
    max_z_ = get_parameter("max_z").as_double();
    voxel_size_ = get_parameter("voxel_size").as_double();
    output_frame_id_ = get_parameter("output_frame_id").as_string();

    // Parameter sanity
    if (min_range_ < 0.0) min_range_ = 0.0;
    if (max_range_ < min_range_) max_range_ = min_range_;
    if (voxel_size_ < 0.0) voxel_size_ = 0.0;

    const auto sensor_qos = rclcpp::SensorDataQoS();
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, sensor_qos,
      std::bind(&CloudFilterCpp::on_cloud, this, std::placeholders::_1));

    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, 10);

    RCLCPP_INFO(get_logger(),
      "cloud_filter_cpp %s -> %s | range[%.2f,%.2f] z[%.2f,%.2f] voxel=%.3f",
      input_topic_.c_str(), output_topic_.c_str(),
      min_range_, max_range_, min_z_, max_z_, voxel_size_);
  }

private:
  void on_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // Defensive: require x/y/z
    if (!has_field(*msg, "x") || !has_field(*msg, "y") || !has_field(*msg, "z")) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "PointCloud2 missing x/y/z fields. Passing empty output.");
      publish_empty_like(*msg);
      return;
    }

    const bool has_intensity = has_field(*msg, "intensity");

    // We'll build output points as float tuples.
    std::vector<float> out_xyz_i;  // packed: x,y,z,(i)
    out_xyz_i.reserve(std::min<std::size_t>(msg->width * msg->height, 200000UL) * (has_intensity ? 4 : 3));

    // Optional voxel dedup
    std::unordered_map<VoxelKey, std::pair<float,float>, VoxelKeyHash> best; 
    // key -> (best_r2, index in out vector start)
    // To keep it simple & safe, we store output index and compare range^2.

    const double vs = voxel_size_;
    const bool use_voxel = (vs > 1e-6);

    sensor_msgs::PointCloud2ConstIterator<float> ix(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iy(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iz(*msg, "z");
    std::optional<sensor_msgs::PointCloud2ConstIterator<float>> ii;
    if (has_intensity) {
      ii.emplace(*msg, "intensity");
    }

    for (; ix != ix.end(); ++ix, ++iy, ++iz) {
      const float x = *ix;
      const float y = *iy;
      const float z = *iz;

      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        if (has_intensity) { ++(*ii); }
        continue;
      }

      const double r = std::sqrt(double(x)*x + double(y)*y + double(z)*z);
      if (!std::isfinite(r) || r < min_range_ || r > max_range_) {
        if (has_intensity) { ++(*ii); }
        continue;
      }
      if (double(z) < min_z_ || double(z) > max_z_) {
        if (has_intensity) { ++(*ii); }
        continue;
      }

      const float intensity = has_intensity ? *(*ii) : 0.0f;
      if (has_intensity) { ++(*ii); }

      if (!use_voxel) {
        out_xyz_i.push_back(x);
        out_xyz_i.push_back(y);
        out_xyz_i.push_back(z);
        if (has_intensity) out_xyz_i.push_back(intensity);
        continue;
      }

      const int64_t vx = static_cast<int64_t>(std::floor(double(x) / vs));
      const int64_t vy = static_cast<int64_t>(std::floor(double(y) / vs));
      const int64_t vz = static_cast<int64_t>(std::floor(double(z) / vs));
      const VoxelKey key{vx, vy, vz};

      const float r2 = x*x + y*y + z*z;

      auto it = best.find(key);
      if (it == best.end()) {
        // store new
        const std::size_t idx = out_xyz_i.size();
        best.emplace(key, std::make_pair(r2, static_cast<float>(idx)));

        out_xyz_i.push_back(x);
        out_xyz_i.push_back(y);
        out_xyz_i.push_back(z);
        if (has_intensity) out_xyz_i.push_back(intensity);
      } else {
        const float best_r2 = it->second.first;
        if (r2 < best_r2) {
          // replace in-place (safe, no realloc)
          const std::size_t idx = static_cast<std::size_t>(it->second.second);
          out_xyz_i[idx + 0] = x;
          out_xyz_i[idx + 1] = y;
          out_xyz_i[idx + 2] = z;
          if (has_intensity) out_xyz_i[idx + 3] = intensity;
          it->second.first = r2;
        }
      }
    }

    publish_cloud(*msg, out_xyz_i, has_intensity);
  }

  void publish_empty_like(const sensor_msgs::msg::PointCloud2 & in)
  {
    std::vector<float> empty;
    publish_cloud(in, empty, has_field(in, "intensity"));
  }

  void publish_cloud(const sensor_msgs::msg::PointCloud2 & in,
                     const std::vector<float> & packed,
                     bool has_intensity)
  {
    sensor_msgs::msg::PointCloud2 out;
    out.header = in.header;
    if (!output_frame_id_.empty()) {
      out.header.frame_id = output_frame_id_;
    }

    out.height = 1;
    const std::size_t stride = has_intensity ? 4 : 3;
    out.width = static_cast<uint32_t>(packed.size() / stride);
    out.is_bigendian = false;
    out.is_dense = true;

    out.fields.clear();
    out.fields.reserve(has_intensity ? 4 : 3);

    sensor_msgs::msg::PointField f;
    f.datatype = sensor_msgs::msg::PointField::FLOAT32;
    f.count = 1;

    f.name = "x"; f.offset = 0;  out.fields.push_back(f);
    f.name = "y"; f.offset = 4;  out.fields.push_back(f);
    f.name = "z"; f.offset = 8;  out.fields.push_back(f);
    if (has_intensity) {
      f.name = "intensity"; f.offset = 12; out.fields.push_back(f);
    }

    out.point_step = has_intensity ? 16 : 12;
    out.row_step = out.point_step * out.width;

    out.data.resize(out.row_step);

    // Copy floats into byte buffer safely
    std::memcpy(out.data.data(), packed.data(), packed.size() * sizeof(float));

    pub_->publish(out);
  }

  static bool has_field(const sensor_msgs::msg::PointCloud2 & msg, const std::string & name)
  {
    for (const auto & f : msg.fields) {
      if (f.name == name) return true;
    }
    return false;
  }

  std::string input_topic_;
  std::string output_topic_;
  std::string output_frame_id_;
  double min_range_;
  double max_range_;
  double min_z_;
  double max_z_;
  double voxel_size_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CloudFilterCpp>());
  rclcpp::shutdown();
  return 0;
}
