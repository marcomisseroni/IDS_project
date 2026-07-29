// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from project_interfaces:msg/Measurement.idl
// generated code does not contain a copyright notice

#ifndef PROJECT_INTERFACES__MSG__DETAIL__MEASUREMENT__BUILDER_HPP_
#define PROJECT_INTERFACES__MSG__DETAIL__MEASUREMENT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "project_interfaces/msg/detail/measurement__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace project_interfaces
{

namespace msg
{

namespace builder
{

class Init_Measurement_dtheta
{
public:
  explicit Init_Measurement_dtheta(::project_interfaces::msg::Measurement & msg)
  : msg_(msg)
  {}
  ::project_interfaces::msg::Measurement dtheta(::project_interfaces::msg::Measurement::_dtheta_type arg)
  {
    msg_.dtheta = std::move(arg);
    return std::move(msg_);
  }

private:
  ::project_interfaces::msg::Measurement msg_;
};

class Init_Measurement_y
{
public:
  explicit Init_Measurement_y(::project_interfaces::msg::Measurement & msg)
  : msg_(msg)
  {}
  Init_Measurement_dtheta y(::project_interfaces::msg::Measurement::_y_type arg)
  {
    msg_.y = std::move(arg);
    return Init_Measurement_dtheta(msg_);
  }

private:
  ::project_interfaces::msg::Measurement msg_;
};

class Init_Measurement_x
{
public:
  explicit Init_Measurement_x(::project_interfaces::msg::Measurement & msg)
  : msg_(msg)
  {}
  Init_Measurement_y x(::project_interfaces::msg::Measurement::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_Measurement_y(msg_);
  }

private:
  ::project_interfaces::msg::Measurement msg_;
};

class Init_Measurement_id_b
{
public:
  explicit Init_Measurement_id_b(::project_interfaces::msg::Measurement & msg)
  : msg_(msg)
  {}
  Init_Measurement_x id_b(::project_interfaces::msg::Measurement::_id_b_type arg)
  {
    msg_.id_b = std::move(arg);
    return Init_Measurement_x(msg_);
  }

private:
  ::project_interfaces::msg::Measurement msg_;
};

class Init_Measurement_id_a
{
public:
  Init_Measurement_id_a()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Measurement_id_b id_a(::project_interfaces::msg::Measurement::_id_a_type arg)
  {
    msg_.id_a = std::move(arg);
    return Init_Measurement_id_b(msg_);
  }

private:
  ::project_interfaces::msg::Measurement msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::project_interfaces::msg::Measurement>()
{
  return project_interfaces::msg::builder::Init_Measurement_id_a();
}

}  // namespace project_interfaces

#endif  // PROJECT_INTERFACES__MSG__DETAIL__MEASUREMENT__BUILDER_HPP_
