// This node subscribes to the super_odometry velodyne_cloud_registered topic,
// applies outlier removal filtering to remove lone floating points,
// and publishes the filtered pointcloud for visualization or further processing.
// Supports both Statistical Outlier Removal (SOR) and Radius Outlier Removal (ROR)
// Also supports ground reflection filtering for indoor environments with reflective floors

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>

class OutlierFilterNode : public rclcpp::Node
{
public:
  OutlierFilterNode() : Node("outlier_filter_node")
  {
    this->declare_parameter("input_topic", "/superodometry/velodyne_cloud_registered");
    this->declare_parameter("output_topic", "/superodometry/velodyne_cloud_outlier_filtered");
    this->declare_parameter("removed_points_topic", "/superodometry/velodyne_cloud_removed_outliers");
    this->declare_parameter("filter_type", "reflection");  // "none", "statistical", "radius", "cascaded", or "reflection"
    this->declare_parameter("sor_mean_k", 50);  // Number of neighbors to analyze (for SOR)
    this->declare_parameter("sor_stddev_mul", 2.0);  // Standard deviation multiplier (for SOR)
    this->declare_parameter("ror_radius", 0.5);  // Search radius in meters (for ROR)
    this->declare_parameter("ror_min_neighbors", 2);  // Minimum neighbors within radius (for ROR)
    this->declare_parameter("reflection_grid_resolution", 0.15);  // XY grid cell size for ground estimation (meters)
    this->declare_parameter("reflection_min_points_for_ground", 3);  // Min points in cell to estimate ground
    this->declare_parameter("reflection_below_ground_threshold", 0.08);  // How far below local ground to filter (meters)
    this->declare_parameter("reflection_search_radius", 0.3);  // Radius to search for local ground points
    
    std::string input_topic = this->get_parameter("input_topic").as_string();
    std::string output_topic = this->get_parameter("output_topic").as_string();
    std::string removed_points_topic = this->get_parameter("removed_points_topic").as_string();
    filter_type_ = this->get_parameter("filter_type").as_string();
    sor_mean_k_ = this->get_parameter("sor_mean_k").as_int();
    sor_stddev_mul_ = this->get_parameter("sor_stddev_mul").as_double();
    ror_radius_ = this->get_parameter("ror_radius").as_double();
    ror_min_neighbors_ = this->get_parameter("ror_min_neighbors").as_int();
    reflection_grid_resolution_ = this->get_parameter("reflection_grid_resolution").as_double();
    reflection_min_points_for_ground_ = this->get_parameter("reflection_min_points_for_ground").as_int();
    reflection_below_ground_threshold_ = this->get_parameter("reflection_below_ground_threshold").as_double();
    reflection_search_radius_ = this->get_parameter("reflection_search_radius").as_double();
    
    RCLCPP_INFO(this->get_logger(), "Outlier Filter Node initialized");
    RCLCPP_INFO(this->get_logger(), "Input topic: %s", input_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "Output topic: %s", output_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "Removed points topic: %s", removed_points_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "Filter type: %s", filter_type_.c_str());
    
    if (filter_type_ == "none") {
      RCLCPP_INFO(this->get_logger(), "No filtering - passthrough mode");
    } else if (filter_type_ == "statistical") {
      RCLCPP_INFO(this->get_logger(), "Statistical Outlier Removal - MeanK: %d, StdDevMul: %.2f", 
                  sor_mean_k_, sor_stddev_mul_);
      sor_.setMeanK(sor_mean_k_);
      sor_.setStddevMulThresh(sor_stddev_mul_);
    } else if (filter_type_ == "radius") {
      RCLCPP_INFO(this->get_logger(), "Radius Outlier Removal - Radius: %.2f m, MinNeighbors: %d", 
                  ror_radius_, ror_min_neighbors_);
      ror_.setRadiusSearch(ror_radius_);
      ror_.setMinNeighborsInRadius(ror_min_neighbors_);
    } else if (filter_type_ == "cascaded") {
      RCLCPP_INFO(this->get_logger(), "Cascaded Filtering (Radius → Statistical)");
      RCLCPP_INFO(this->get_logger(), "  Stage 1 - Radius: %.2f m, MinNeighbors: %d", 
                  ror_radius_, ror_min_neighbors_);
      RCLCPP_INFO(this->get_logger(), "  Stage 2 - Statistical: MeanK: %d, StdDevMul: %.2f", 
                  sor_mean_k_, sor_stddev_mul_);
      ror_.setRadiusSearch(ror_radius_);
      ror_.setMinNeighborsInRadius(ror_min_neighbors_);
      sor_.setMeanK(sor_mean_k_);
      sor_.setStddevMulThresh(sor_stddev_mul_);
    } else if (filter_type_ == "reflection") {
      RCLCPP_INFO(this->get_logger(), "Reflection Filter (ground reflection removal for reflective floors)");
      RCLCPP_INFO(this->get_logger(), "  Grid resolution: %.2f m, Below threshold: %.2f m, Search radius: %.2f m",
                  reflection_grid_resolution_, reflection_below_ground_threshold_, reflection_search_radius_);
    } else {
      RCLCPP_ERROR(this->get_logger(), "Unknown filter_type: %s. Use 'none', 'statistical', 'radius', 'cascaded', or 'reflection'", 
                   filter_type_.c_str());
      rclcpp::shutdown();
    }
    
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic, 10,
      std::bind(&OutlierFilterNode::pointcloudCallback, this, std::placeholders::_1));
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic, 10);
    pub_removed_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(removed_points_topic, 10);
    
    // For keeping track of statistics
    total_received_ = 0;
    total_processed_ = 0;
    total_reflection_filtered_ = 0;
  }

private:
  /**
   * Ground reflection filter that removes points below the local ground surface.
   * Works by:
   * 1. Building a 2D grid (XY) and finding the lowest points in each cell as ground candidates
   * 2. For each point, estimate local ground height from nearby cells
   * 3. Remove points that are significantly below the local ground estimate
   * 
   * This handles non-flat terrain because ground is estimated locally, not globally.
   */
  void applyReflectionFilter(
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in,
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_out,
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_removed)
  {
    if (cloud_in->empty()) {
      *cloud_out = *cloud_in;
      return;
    }
    
    // Step 1: Build a 2D grid map to estimate local ground height
    // Key: (grid_x, grid_y) -> vector of z values in that cell
    std::map<std::pair<int, int>, std::vector<float>> grid_z_values;
    std::map<std::pair<int, int>, float> grid_ground_z;  // Estimated ground Z for each cell
    
    // Compute grid indices for all points and collect z values
    for (const auto& pt : cloud_in->points) {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
        continue;
      }
      int gx = static_cast<int>(std::floor(pt.x / reflection_grid_resolution_));
      int gy = static_cast<int>(std::floor(pt.y / reflection_grid_resolution_));
      grid_z_values[{gx, gy}].push_back(pt.z);
    }
    
    // Step 2: For each cell, estimate ground as a low percentile of z values
    // Using percentile instead of min to be robust to noise
    for (auto& [key, z_vals] : grid_z_values) {
      if (z_vals.size() < static_cast<size_t>(reflection_min_points_for_ground_)) {
        continue;  // Not enough points to estimate ground
      }
      
      std::sort(z_vals.begin(), z_vals.end());
      
      // Use ~10th percentile as ground estimate (robust to outliers below ground)
      size_t ground_idx = std::max(static_cast<size_t>(0), 
                                   static_cast<size_t>(z_vals.size() * 0.1));
      grid_ground_z[key] = z_vals[ground_idx];
    }
    
    // Step 3: For each point, check if it's below local ground
    cloud_out->points.clear();
    cloud_out->points.reserve(cloud_in->points.size());
    cloud_removed->points.clear();
    
    int search_cells = static_cast<int>(std::ceil(reflection_search_radius_ / reflection_grid_resolution_));
    size_t reflection_count = 0;
    
    for (const auto& pt : cloud_in->points) {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
        cloud_removed->points.push_back(pt);
        continue;
      }
      
      int gx = static_cast<int>(std::floor(pt.x / reflection_grid_resolution_));
      int gy = static_cast<int>(std::floor(pt.y / reflection_grid_resolution_));
      
      // Gather ground estimates from nearby cells
      std::vector<float> nearby_ground_z;
      for (int dx = -search_cells; dx <= search_cells; ++dx) {
        for (int dy = -search_cells; dy <= search_cells; ++dy) {
          auto it = grid_ground_z.find({gx + dx, gy + dy});
          if (it != grid_ground_z.end()) {
            nearby_ground_z.push_back(it->second);
          }
        }
      }
      
      bool is_reflection = false;
      
      if (!nearby_ground_z.empty()) {
        // Use median of nearby ground estimates for robustness
        std::sort(nearby_ground_z.begin(), nearby_ground_z.end());
        float local_ground = nearby_ground_z[nearby_ground_z.size() / 2];
        
        // Check if point is below ground threshold
        if (pt.z < local_ground - reflection_below_ground_threshold_) {
          is_reflection = true;
        }
      }
      
      if (is_reflection) {
        cloud_removed->points.push_back(pt);
        reflection_count++;
      } else {
        cloud_out->points.push_back(pt);
      }
    }
    
    cloud_out->width = cloud_out->points.size();
    cloud_out->height = 1;
    cloud_out->is_dense = true;
    
    cloud_removed->width = cloud_removed->points.size();
    cloud_removed->height = 1;
    cloud_removed->is_dense = true;
    
    total_reflection_filtered_ += reflection_count;
  }
  void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    total_received_++;
    
    auto start_time = this->now();
    
    // Convert ROS PointCloud2 to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_removed(new pcl::PointCloud<pcl::PointXYZI>);
    
    pcl::fromROSMsg(*msg, *cloud);
    
    size_t original_size = cloud->points.size();
    
    // Apply filtering based on filter type
    if (filter_type_ == "reflection") {
      // Reflection filter only
      applyReflectionFilter(cloud, cloud_filtered, cloud_removed);
    } else if (filter_type_ == "none") {
      // No filtering - just pass through
      *cloud_filtered = *cloud;
      // Empty removed cloud
      cloud_removed->points.clear();
      cloud_removed->header = cloud->header;
      cloud_removed->width = 0;
      cloud_removed->height = 1;
    } else if (filter_type_ == "statistical") {
      // Statistical Outlier Removal only
      sor_.setInputCloud(cloud);
      sor_.setNegative(false);
      sor_.filter(*cloud_filtered);
      sor_.setNegative(true);
      sor_.filter(*cloud_removed);
    } else if (filter_type_ == "radius") {
      // Radius Outlier Removal only
      ror_.setInputCloud(cloud);
      ror_.setNegative(false);
      ror_.filter(*cloud_filtered);
      ror_.setNegative(true);
      ror_.filter(*cloud_removed);
    } else if (filter_type_ == "cascaded") {
      // Cascaded: Radius first, then Statistical
      pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_intermediate(new pcl::PointCloud<pcl::PointXYZI>);
      pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_removed_stage1(new pcl::PointCloud<pcl::PointXYZI>);
      pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_removed_stage2(new pcl::PointCloud<pcl::PointXYZI>);
      
      // Stage 1: Radius Outlier Removal
      ror_.setInputCloud(cloud);
      ror_.setNegative(false);
      ror_.filter(*cloud_intermediate);
      ror_.setNegative(true);
      ror_.filter(*cloud_removed_stage1);
      
      size_t after_radius = cloud_intermediate->points.size();
      
      // Stage 2: Statistical Outlier Removal on radius-filtered cloud
      sor_.setInputCloud(cloud_intermediate);
      sor_.setNegative(false);
      sor_.filter(*cloud_filtered);
      sor_.setNegative(true);
      sor_.filter(*cloud_removed_stage2);
      
      // Combine removed points from both stages
      *cloud_removed = *cloud_removed_stage1 + *cloud_removed_stage2;
      
      // Debug output for cascaded filtering
      if (total_processed_ % 25 == 0) {
        RCLCPP_INFO(this->get_logger(), 
                    "Cascaded: %zu -> %zu (radius) -> %zu (statistical) points",
                    original_size, after_radius, cloud_filtered->points.size());
      }
    }
    
    size_t filtered_size = cloud_filtered->points.size();
    size_t removed_size = cloud_removed->points.size();
    
    // Convert back to ROS PointCloud2
    sensor_msgs::msg::PointCloud2 output_msg;
    pcl::toROSMsg(*cloud_filtered, output_msg);
    output_msg.header = msg->header;
    
    pub_->publish(output_msg);
    
    // Publish the removed points
    sensor_msgs::msg::PointCloud2 removed_msg;
    pcl::toROSMsg(*cloud_removed, removed_msg);
    removed_msg.header = msg->header;
    pub_removed_->publish(removed_msg);
    
    total_processed_++;
    
    auto end_time = this->now();
    double processing_time_ms = (end_time - start_time).seconds() * 1000.0;
    
    // Log statistics periodically
    if (total_processed_ % 25 == 0) {
      double removal_rate = 100.0 * (1.0 - static_cast<double>(filtered_size) / original_size);
      RCLCPP_INFO(this->get_logger(), 
                  "Processed %ld clouds | Latest: %zu -> %zu points (%.1f%% removed, %zu filtered) | Time: %.2f ms",
                  total_processed_, original_size, filtered_size, removal_rate, removed_size, processing_time_ms);
    }
  }
  
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_removed_;
  
  std::string filter_type_;
  int sor_mean_k_;
  double sor_stddev_mul_;
  double ror_radius_;
  int ror_min_neighbors_;
  
  // Reflection filter parameters
  double reflection_grid_resolution_;
  int reflection_min_points_for_ground_;
  double reflection_below_ground_threshold_;
  double reflection_search_radius_;
  
  // Filter objects (reuse for performance)
  pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor_;
  pcl::RadiusOutlierRemoval<pcl::PointXYZI> ror_;
  
  // For keeping track of statistics
  size_t total_received_;
  size_t total_processed_;
  size_t total_reflection_filtered_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<OutlierFilterNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
