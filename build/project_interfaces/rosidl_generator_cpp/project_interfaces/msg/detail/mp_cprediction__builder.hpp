// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from project_interfaces:msg/MPCprediction.idl
// generated code does not contain a copyright notice

#ifndef PROJECT_INTERFACES__MSG__DETAIL__MP_CPREDICTION__BUILDER_HPP_
#define PROJECT_INTERFACES__MSG__DETAIL__MP_CPREDICTION__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "project_interfaces/msg/detail/mp_cprediction__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace project_interfaces
{

namespace msg
{

namespace builder
{

class Init_MPCprediction_theta
{
public:
  explicit Init_MPCprediction_theta(::project_interfaces::msg::MPCprediction & msg)
  : msg_(msg)
  {}
  ::project_interfaces::msg::MPCprediction theta(::project_interfaces::msg::MPCprediction::_theta_type arg)
  {
    msg_.theta = std::move(arg);
    return std::move(msg_);
  }

private:
  ::project_interfaces::msg::MPCprediction msg_;
};

class Init_MPCprediction_y
{
public:
  explicit Init_MPCprediction_y(::project_interfaces::msg::MPCprediction & msg)
  : msg_(msg)
  {}
  Init_MPCprediction_theta y(::project_interfaces::msg::MPCprediction::_y_type arg)
  {
    msg_.y = std::move(arg);
    return Init_MPCprediction_theta(msg_);
  }

private:
  ::project_interfaces::msg::MPCprediction msg_;
};

class Init_MPCprediction_x
{
public:
  Init_MPCprediction_x()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MPCprediction_y x(::project_interfaces::msg::MPCprediction::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_MPCprediction_y(msg_);
  }

private:
  ::project_interfaces::msg::MPCprediction msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::project_interfaces::msg::MPCprediction>()
{
  return project_interfaces::msg::builder::Init_MPCprediction_x();
}

}  // namespace project_interfaces

#endif  // PROJECT_INTERFACES__MSG__DETAIL__MP_CPREDICTION__BUILDER_HPP_
