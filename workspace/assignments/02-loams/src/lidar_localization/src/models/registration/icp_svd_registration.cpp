#include "lidar_localization/models/registration/icp_svd_registration.hpp"

#include "glog/logging.h"
#include <pcl/common/transforms.h>

namespace lidar_localization {

ICPSVDRegistration::ICPSVDRegistration(const YAML::Node& _node):
    input_target_kdtree_(new pcl::KdTreeFLANN<pcl::PointXYZ>()) {
    
    float max_corr_dist = _node["max_corr_dist"].as<float>();
    float trans_eps = _node["trans_eps"].as<float>();
    float euc_fitness_eps = _node["euc_fitness_eps"].as<float>();
    int max_iter = _node["max_iter"].as<int>();

    SetRegistrationParam(max_corr_dist, trans_eps, euc_fitness_eps, max_iter);


}

ICPSVDRegistration::ICPSVDRegistration(float _max_corr_dist, float _trans_eps, float _euc_fitness_eps, int _max_iter):
    input_target_kdtree_(new pcl::KdTreeFLANN<pcl::PointXYZ>()) {
    
    SetRegistrationParam(_max_corr_dist, _trans_eps, _euc_fitness_eps, _max_iter);
}


bool ICPSVDRegistration::SetInputTarget(const CloudData::CLOUD_PTR& _input_target) {
    input_target_ = _input_target;
    input_target_kdtree_->setInputCloud(input_target_);
    return true;
}
    
bool ICPSVDRegistration::ScanMatch(const CloudData::CLOUD_PTR& _input_source,
                    const Eigen::Matrix4f& _predict_pose,
                    CloudData::CLOUD_PTR& _result_clould_ptr,
                    Eigen::Matrix4f& _result_pose) {

    input_source_ = _input_source;
    CloudData::CLOUD_PTR transformed_input_source(new CloudData::CLOUD());

    // preprocess the source cloud, set an initial value
    pcl::transformPointCloud(*input_source_, *transformed_input_source, _predict_pose);
    
    // init estimation
    transformation_.setIdentity();

    // start iteration
    int cur_iter = 0;
    while(cur_iter < max_iter_) {
        
        CloudData::CLOUD_PTR cur_input_source(new CloudData::CLOUD());
        pcl::transformPointCloud(*transformed_input_source, *cur_input_source, transformation_);

        // get Nearest Neighbor
        std::vector<Eigen::Vector3f> target_points;
        std::vector<Eigen::Vector3f> source_points;

        // do not have enough correspondence, break
        if(GetNearestNeighbor(cur_input_source, target_points, source_points) < 3) { break; }

        // update current transformation
        Eigen::Matrix4f delta_transformation;
        GetTransform(target_points, source_points, delta_transformation);

        // check if the update is small enough
        if(IsConverged(delta_transformation, trans_eps_)) { break; }

        transformation_ = delta_transformation * transformation_;

        cur_iter ++;
    }

    _result_pose = transformation_ * _predict_pose;
    pcl::transformPointCloud(*_input_source, *_result_clould_ptr, _result_pose);

    return true;

}

bool ICPSVDRegistration::SetRegistrationParam(float _max_corr_dist, float _trans_eps, float _euc_fitness_eps, int _max_iter) {
    max_corr_dist_ = _max_corr_dist;
    trans_eps_ = _trans_eps;
    euc_fitness_eps_ = _euc_fitness_eps;
    max_iter_ = _max_iter;

    LOG(INFO) << "ICP SVD params:" << std::endl
              << "max_corr_dist: " << max_corr_dist_ << ", "
              << "trans_eps: " << trans_eps_ << ", "
              << "euc_fitness_eps: " << euc_fitness_eps_ << ", "
              << "max_iter: " << max_iter_ 
              << std::endl << std::endl;

    return true;
}

size_t ICPSVDRegistration::GetNearestNeighbor(
    const CloudData::CLOUD_PTR& _input_source,
    std::vector<Eigen::Vector3f>& _target_points,
    std::vector<Eigen::Vector3f>& _source_points) {

    float MAX_DIST = max_corr_dist_*max_corr_dist_;

    size_t corr_num = 0;
    for(size_t i=0; i<_input_source->points.size(); ++i) {
        std::vector<int> corr_idx;
        std::vector<float> corr_dist;

        // the value of corr_dist is the squred value
        input_target_kdtree_->nearestKSearch(_input_source->at(i), 1, corr_idx, corr_dist);
        
        if(corr_dist[0] > MAX_DIST) { continue; }

        Eigen::Vector3f target_point(input_target_->at(corr_idx[0]).x, input_target_->at(corr_idx[0]).y, input_target_->at(corr_idx[0]).z);
        Eigen::Vector3f source_point(_input_source->at(i).x, _input_source->at(i).y, _input_source->at(i).z);

        _target_points.push_back(target_point);
        _source_points.push_back(source_point);

        corr_num ++;
    }

    return corr_num;
}

void ICPSVDRegistration::GetTransform(const std::vector<Eigen::Vector3f>& _target_points, const std::vector<Eigen::Vector3f>& _source_points, Eigen::Matrix4f& _transform) {
    int N = _target_points.size();
    Eigen::Vector3f mu_target = Eigen::Vector3f::Zero();
    Eigen::Vector3f mu_source = Eigen::Vector3f::Zero();

    for(int i=0; i<N; ++i) {
        mu_target += _target_points[i];
        mu_source += _source_points[i];
    }
    mu_target /= N;
    mu_source /= N;

    // build H
    Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
    for(int i=0; i<N; ++i) {
        H += (_source_points[i] - mu_source) * (_target_points[i] - mu_target).transpose();
    }

    // solve R and t
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3f R = svd.matrixV() * svd.matrixU().transpose();
    Eigen::Vector3f t = mu_target - R * mu_source;

    _transform.setIdentity();
    _transform.block<3,3>(0,0) = R;
    _transform.block<3,1>(0,3) = t;
}

bool ICPSVDRegistration::IsConverged(const Eigen::Matrix4f _transform, const float _trans_eps) {
    
    float translation_norm = _transform.block<3,1>(0,3).norm();
    float rotation_angle = fabs(acos((_transform.block<3,3>(0,0).trace() - 1.0) / 2.0));
    
    return translation_norm<_trans_eps && rotation_angle<_trans_eps;
}

}