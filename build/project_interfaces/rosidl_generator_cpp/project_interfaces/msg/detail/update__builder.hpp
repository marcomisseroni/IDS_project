// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from project_interfaces:msg/Update.idl
// generated code does not contain a copyright notice

#ifndef PROJECT_INTERFACES__MSG__DETAIL__UPDATE__BUILDER_HPP_
#define PROJECT_INTERFACES__MSG__DETAIL__UPDATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "project_interfaces/msg/detail/update__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace project_interfaces
{

namespace msg
{

namespace builder
{

class Init_Update_w2
{
public:
  explicit Init_Update_w2(::project_interfaces::msg::Update & msg)
  : msg_(msg)
  {}
  ::project_interfaces::msg::Update w2(::project_interfaces::msg::Update::_w2_type arg)
  {
    msg_.w2 = std::move(arg);
    return std::move(msg_);
  }

private:
  ::project_interfaces::msg::Update msg_;
};

class Init_Update_w1
{
public:
  explicit Init_Update_w1(::project_interfaces::msg::Update & msg)
  : msg_(msg)
  {}
  Init_Update_w2 w1(::project_interfaces::msg::Update::_w1_type arg)
  {
    msg_.w1 = std::move(arg);
    return Init_Update_w2(msg_);
  }

private:
  ::project_interfaces::msg::Update msg_;
};

class Init_Update_gamma_b
{
public:
  explicit Init_Update_gamma_b(::project_interfaces::msg::Update & msg)
  : msg_(msg)
  {}
  Init_Update_w1 gamma_b(::project_interfaces::msg::Update::_gamma_b_type arg)
  {
    msg_.gamma_b = std::move(arg);
    return Init_Update_w1(msg_);
  }

private:
  ::project_interfaces::msg::Update msg_;
};

class Init_Update_gamma_a
{
public:
  explicit Init_Update_gamma_a(::project_interfaces::msg::Update & msg)
  : msg_(msg)
  {}
  Init_Update_gamma_b gamma_a(::project_interfaces::msg::Update::_gamma_a_type arg)
  {
    msg_.gamma_a = std::move(arg);
    return Init_Update_gamma_b(msg_);
  }

private:
  ::project_interfaces::msg::Update msg_;
};

class Init_Update_ra
{
public:
  explicit Init_Update_ra(::project_interfaces::msg::Update & msg)
  : msg_(msg)
  {}
  Init_Update_gamma_a ra(::project_interfaces::msg::Update::_ra_type arg)
  {
    msg_.ra = std::move(arg);
    return Init_Update_gamma_a(msg_);
  }

private:
  ::project_interfaces::msg::Update msg_;
};

class Init_Update_dim_b
{
public:
  explicit Init_Update_dim_b(::project_interfaces::msg::Update & msg)
  : msg_(msg)
  {}
  Init_Update_ra dim_b(::project_interfaces::msg::Update::_dim_b_type arg)
  {
    msg_.dim_b = std::move(arg);
    return Init_Update_ra(msg_);
  }

private:
  ::project_interfaces::msg::Update msg_;
};

class Init_Update_dim_a
{
public:
  explicit Init_Update_dim_a(::project_interfaces::msg::Update & msg)
  : msg_(msg)
  {}
  Init_Update_dim_b dim_a(::project_interfaces::msg::Update::_dim_a_type arg)
  {
    msg_.dim_a = std::move(arg);
    return Init_Update_dim_b(msg_);
  }

private:
  ::project_interfaces::msg::Update msg_;
};

class Init_Update_id_b
{
public:
  explicit Init_Update_id_b(::project_interfaces::msg::Update & msg)
  : msg_(msg)
  {}
  Init_Update_dim_a id_b(::project_interfaces::msg::Update::_id_b_type arg)
  {
    msg_.id_b = std::move(arg);
    return Init_Update_dim_a(msg_);
  }

private:
  ::project_interfaces::msg::Update msg_;
};

class Init_Update_id_a
{
public:
  Init_Update_id_a()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Update_id_b id_a(::project_interfaces::msg::Update::_id_a_type arg)
  {
    msg_.id_a = std::move(arg);
    return Init_Update_id_b(msg_);
  }

private:
  ::project_interfaces::msg::Update msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::project_interfaces::msg::Update>()
{
  return project_interfaces::msg::builder::Init_Update_id_a();
}

}  // namespace project_interfaces

#endif  // PROJECT_INTERFACES__MSG__DETAIL__UPDATE__BUILDER_HPP_
