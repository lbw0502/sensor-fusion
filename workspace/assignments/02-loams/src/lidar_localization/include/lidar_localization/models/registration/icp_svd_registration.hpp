#ifndef LIDAR_LOCALIZATION_MODELS_REGISTRATION_ICP_SVD_REGISTRATION_HPP_
#define LIDAR_LOCALIZATION_MODELS_REGISTRATION_ICP_SVD_REGISTRATION_HPP_

#include "lidar_localization/models/registration/registration_interface.hpp"
#include <pcl/kdtree/kdtree_flann.h>

namespace lidar_localization {
class ICPSVDRegistration: public RegistrationInterface {
public:
    ICPSVDRegistration(const YAML::Node& _node);

    ICPSVDRegistration(
        float _max_corr_dist,
        float _trans_eps,
        float _euc_fitness_eps,
        int _max_iter
    );
    
    bool SetInputTarget(const CloudData::CLOUD_PTR& _input_target) override;

    bool ScanMatch(const CloudData::CLOUD_PTR& _input_source,
                    const Eigen::Matrix4f& _predict_pose,
                    CloudData::CLOUD_PTR& result_clould_ptr,
                    Eigen::Matrix4f& result_pose) override;

private:
    bool SetRegistrationParam(
        float _max_corr_dist, 
        float _trans_eps, 
        float _euc_fitness_eps, 
        int _max_iter);

    size_t GetNearestNeighbor(
        const CloudData::CLOUD_PTR& _input_source,
        std::vector<Eigen::Vector3f>& _target_points,
        std::vector<Eigen::Vector3f>& _souce_points);

    void GetTransform(
        const std::vector<Eigen::Vector3f>& _target_points,
        const std::vector<Eigen::Vector3f>& _source_points,
        Eigen::Matrix4f& _transform);

    bool IsConverged(const Eigen::Matrix4f _transform, const float _trans_eps);


private:
    float max_corr_dist_;
    float trans_eps_;
    float euc_fitness_eps_;
    int max_iter_;

    CloudData::CLOUD_PTR input_source_;
    CloudData::CLOUD_PTR input_target_;
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr input_target_kdtree_;

    Eigen::Matrix4f transformation_;
};

}
#endif