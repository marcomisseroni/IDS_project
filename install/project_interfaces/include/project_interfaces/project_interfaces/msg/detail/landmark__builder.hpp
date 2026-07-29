// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from project_interfaces:msg/Landmark.idl
// generated code does not contain a copyright notice

#ifndef PROJECT_INTERFACES__MSG__DETAIL__LANDMARK__BUILDER_HPP_
#define PROJECT_INTERFACES__MSG__DETAIL__LANDMARK__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "project_interfaces/msg/detail/landmark__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace project_interfaces
{

namespace msg
{

namespace builder
{

class Init_Landmark_p
{
public:
  explicit Init_Landmark_p(::project_interfaces::msg::Landmark & msg)
  : msg_(msg)
  {}
  ::project_interfaces::msg::Landmark p(::project_interfaces::msg::Landmark::_p_type arg)
  {
    msg_.p = std::move(arg);
    return std::move(msg_);
  }

private:
  ::project_interfaces::msg::Landmark msg_;
};

class Init_Landmark_phi
{
public:
  explicit Init_Landmark_phi(::project_interfaces::msg::Landmark & msg)
  : msg_(msg)
  {}
  Init_Landmark_p phi(::project_interfaces::msg::Landmark::_phi_type arg)
  {
    msg_.phi = std::move(arg);
    return Init_Landmark_p(msg_);
  }

private:
  ::project_interfaces::msg::Landmark msg_;
};

class Init_Landmark_state
{
public:
  explicit Init_Landmark_state(::project_interfaces::msg::Landmark & msg)
  : msg_(msg)
  {}
  Init_Landmark_phi state(::project_interfaces::msg::Landmark::_state_type arg)
  {
    msg_.state = std::move(arg);
    return Init_Landmark_phi(msg_);
  }

private:
  ::project_interfaces::msg::Landmark msg_;
};

class Init_Landmark_id_b
{
public:
  explicit Init_Landmark_id_b(::project_interfaces::msg::Landmark & msg)
  : msg_(msg)
  {}
  Init_Landmark_state id_b(::project_interfaces::msg::Landmark::_id_b_type arg)
  {
    msg_.id_b = std::move(arg);
    return Init_Landmark_state(msg_);
  }

private:
  ::project_interfaces::msg::Landmark msg_;
};

class Init_Landmark_id_a
{
public:
  explicit Init_Landmark_id_a(::project_interfaces::msg::Landmark & msg)
  : msg_(msg)
  {}
  Init_Landmark_id_b id_a(::project_interfaces::msg::Landmark::_id_a_type arg)
  {
    msg_.id_a = std::move(arg);
    return Init_Landmark_id_b(msg_);
  }

private:
  ::project_interfaces::msg::Landmark msg_;
};

class Init_Landmark_dim
{
public:
  Init_Landmark_dim()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Landmark_id_a dim(::project_interfaces::msg::Landmark::_dim_type arg)
  {
    msg_.dim = std::move(arg);
    return Init_Landmark_id_a(msg_);
  }

private:
  ::project_interfaces::msg::Landmark msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::project_interfaces::msg::Landmark>()
{
  return project_interfaces::msg::builder::Init_Landmark_dim();
}

}  // namespace project_interfaces

#endif  // PROJECT_INTERFACES__MSG__DETAIL__LANDMARK__BUILDER_HPP_
