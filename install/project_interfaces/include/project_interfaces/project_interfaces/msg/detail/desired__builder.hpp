// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from project_interfaces:msg/Desired.idl
// generated code does not contain a copyright notice

#ifndef PROJECT_INTERFACES__MSG__DETAIL__DESIRED__BUILDER_HPP_
#define PROJECT_INTERFACES__MSG__DETAIL__DESIRED__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "project_interfaces/msg/detail/desired__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace project_interfaces
{

namespace msg
{

namespace builder
{

class Init_Desired_y2
{
public:
  explicit Init_Desired_y2(::project_interfaces::msg::Desired & msg)
  : msg_(msg)
  {}
  ::project_interfaces::msg::Desired y2(::project_interfaces::msg::Desired::_y2_type arg)
  {
    msg_.y2 = std::move(arg);
    return std::move(msg_);
  }

private:
  ::project_interfaces::msg::Desired msg_;
};

class Init_Desired_x2
{
public:
  explicit Init_Desired_x2(::project_interfaces::msg::Desired & msg)
  : msg_(msg)
  {}
  Init_Desired_y2 x2(::project_interfaces::msg::Desired::_x2_type arg)
  {
    msg_.x2 = std::move(arg);
    return Init_Desired_y2(msg_);
  }

private:
  ::project_interfaces::msg::Desired msg_;
};

class Init_Desired_y1
{
public:
  explicit Init_Desired_y1(::project_interfaces::msg::Desired & msg)
  : msg_(msg)
  {}
  Init_Desired_x2 y1(::project_interfaces::msg::Desired::_y1_type arg)
  {
    msg_.y1 = std::move(arg);
    return Init_Desired_x2(msg_);
  }

private:
  ::project_interfaces::msg::Desired msg_;
};

class Init_Desired_x1
{
public:
  explicit Init_Desired_x1(::project_interfaces::msg::Desired & msg)
  : msg_(msg)
  {}
  Init_Desired_y1 x1(::project_interfaces::msg::Desired::_x1_type arg)
  {
    msg_.x1 = std::move(arg);
    return Init_Desired_y1(msg_);
  }

private:
  ::project_interfaces::msg::Desired msg_;
};

class Init_Desired_y0
{
public:
  explicit Init_Desired_y0(::project_interfaces::msg::Desired & msg)
  : msg_(msg)
  {}
  Init_Desired_x1 y0(::project_interfaces::msg::Desired::_y0_type arg)
  {
    msg_.y0 = std::move(arg);
    return Init_Desired_x1(msg_);
  }

private:
  ::project_interfaces::msg::Desired msg_;
};

class Init_Desired_x0
{
public:
  Init_Desired_x0()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Desired_y0 x0(::project_interfaces::msg::Desired::_x0_type arg)
  {
    msg_.x0 = std::move(arg);
    return Init_Desired_y0(msg_);
  }

private:
  ::project_interfaces::msg::Desired msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::project_interfaces::msg::Desired>()
{
  return project_interfaces::msg::builder::Init_Desired_x0();
}

}  // namespace project_interfaces

#endif  // PROJECT_INTERFACES__MSG__DETAIL__DESIRED__BUILDER_HPP_
