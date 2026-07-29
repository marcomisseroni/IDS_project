// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from project_interfaces:msg/State.idl
// generated code does not contain a copyright notice

#ifndef PROJECT_INTERFACES__MSG__DETAIL__STATE__BUILDER_HPP_
#define PROJECT_INTERFACES__MSG__DETAIL__STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "project_interfaces/msg/detail/state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace project_interfaces
{

namespace msg
{

namespace builder
{

class Init_State_theta
{
public:
  explicit Init_State_theta(::project_interfaces::msg::State & msg)
  : msg_(msg)
  {}
  ::project_interfaces::msg::State theta(::project_interfaces::msg::State::_theta_type arg)
  {
    msg_.theta = std::move(arg);
    return std::move(msg_);
  }

private:
  ::project_interfaces::msg::State msg_;
};

class Init_State_y
{
public:
  explicit Init_State_y(::project_interfaces::msg::State & msg)
  : msg_(msg)
  {}
  Init_State_theta y(::project_interfaces::msg::State::_y_type arg)
  {
    msg_.y = std::move(arg);
    return Init_State_theta(msg_);
  }

private:
  ::project_interfaces::msg::State msg_;
};

class Init_State_x
{
public:
  explicit Init_State_x(::project_interfaces::msg::State & msg)
  : msg_(msg)
  {}
  Init_State_y x(::project_interfaces::msg::State::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_State_y(msg_);
  }

private:
  ::project_interfaces::msg::State msg_;
};

class Init_State_id
{
public:
  Init_State_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_State_x id(::project_interfaces::msg::State::_id_type arg)
  {
    msg_.id = std::move(arg);
    return Init_State_x(msg_);
  }

private:
  ::project_interfaces::msg::State msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::project_interfaces::msg::State>()
{
  return project_interfaces::msg::builder::Init_State_id();
}

}  // namespace project_interfaces

#endif  // PROJECT_INTERFACES__MSG__DETAIL__STATE__BUILDER_HPP_
