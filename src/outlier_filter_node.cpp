// This node subscribes to the super_odometry velodyne_cloud_registered topic,
// applies outlier removal filtering to remove lone floating points,
// and publishes the filtered pointcloud for visualization or further processing.
// Supports both Statistical Outlier Removal (SOR) and Radius Outlier Removal (ROR)

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
    this->declare_parameter("filter_type", "radius");  // "statistical", "radius", or "cascaded"
    this->declare_parameter("sor_mean_k", 50);  // Number of neighbors to analyze (for SOR)
    this->declare_parameter("sor_stddev_mul", 2.0);  // Standard deviation multiplier (for SOR)
    this->declare_parameter("ror_radius", 0.5);  // Search radius in meters (for ROR)
    this->declare_parameter("ror_min_neighbors", 2);  // Minimum neighbors within radius (for ROR)
    
    std::string input_topic = this->get_parameter("input_topic").as_string();
    std::string output_topic = this->get_parameter("output_topic").as_string();
    std::string removed_points_topic = this->get_parameter("removed_points_topic").as_string();
    filter_type_ = this->get_parameter("filter_type").as_string();
    sor_mean_k_ = this->get_parameter("sor_mean_k").as_int();
    sor_stddev_mul_ = this->get_parameter("sor_stddev_mul").as_double();
    ror_radius_ = this->get_parameter("ror_radius").as_double();
    ror_min_neighbors_ = this->get_parameter("ror_min_neighbors").as_int();
    
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
    } else {
      RCLCPP_ERROR(this->get_logger(), "Unknown filter_type: %s. Use 'none', 'statistical', 'radius', or 'cascaded'", 
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
  }

private:
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
    
    // Apply outlier removal based on filter type
    if (filter_type_ == "none") {
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
                  "Processed %ld clouds | Latest: %zu -> %zu points (%.1f%% removed, %zu outliers) | Time: %.2f ms",
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
  
  // Filter objects (reuse for performance)
  pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor_;
  pcl::RadiusOutlierRemoval<pcl::PointXYZI> ror_;
  
  // For keeping track of statistics
  size_t total_received_;
  size_t total_processed_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<OutlierFilterNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
