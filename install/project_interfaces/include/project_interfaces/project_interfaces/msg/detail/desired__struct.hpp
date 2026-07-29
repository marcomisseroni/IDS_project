// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from project_interfaces:msg/Desired.idl
// generated code does not contain a copyright notice

#ifndef PROJECT_INTERFACES__MSG__DETAIL__DESIRED__STRUCT_HPP_
#define PROJECT_INTERFACES__MSG__DETAIL__DESIRED__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__project_interfaces__msg__Desired __attribute__((deprecated))
#else
# define DEPRECATED__project_interfaces__msg__Desired __declspec(deprecated)
#endif

namespace project_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Desired_
{
  using Type = Desired_<ContainerAllocator>;

  explicit Desired_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->x0 = 0.0;
      this->y0 = 0.0;
      this->x1 = 0.0;
      this->y1 = 0.0;
      this->x2 = 0.0;
      this->y2 = 0.0;
    }
  }

  explicit Desired_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->x0 = 0.0;
      this->y0 = 0.0;
      this->x1 = 0.0;
      this->y1 = 0.0;
      this->x2 = 0.0;
      this->y2 = 0.0;
    }
  }

  // field types and members
  using _x0_type =
    double;
  _x0_type x0;
  using _y0_type =
    double;
  _y0_type y0;
  using _x1_type =
    double;
  _x1_type x1;
  using _y1_type =
    double;
  _y1_type y1;
  using _x2_type =
    double;
  _x2_type x2;
  using _y2_type =
    double;
  _y2_type y2;

  // setters for named parameter idiom
  Type & set__x0(
    const double & _arg)
  {
    this->x0 = _arg;
    return *this;
  }
  Type & set__y0(
    const double & _arg)
  {
    this->y0 = _arg;
    return *this;
  }
  Type & set__x1(
    const double & _arg)
  {
    this->x1 = _arg;
    return *this;
  }
  Type & set__y1(
    const double & _arg)
  {
    this->y1 = _arg;
    return *this;
  }
  Type & set__x2(
    const double & _arg)
  {
    this->x2 = _arg;
    return *this;
  }
  Type & set__y2(
    const double & _arg)
  {
    this->y2 = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    project_interfaces::msg::Desired_<ContainerAllocator> *;
  using ConstRawPtr =
    const project_interfaces::msg::Desired_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<project_interfaces::msg::Desired_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<project_interfaces::msg::Desired_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      project_interfaces::msg::Desired_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<project_interfaces::msg::Desired_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      project_interfaces::msg::Desired_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<project_interfaces::msg::Desired_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<project_interfaces::msg::Desired_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<project_interfaces::msg::Desired_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__project_interfaces__msg__Desired
    std::shared_ptr<project_interfaces::msg::Desired_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__project_interfaces__msg__Desired
    std::shared_ptr<project_interfaces::msg::Desired_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Desired_ & other) const
  {
    if (this->x0 != other.x0) {
      return false;
    }
    if (this->y0 != other.y0) {
      return false;
    }
    if (this->x1 != other.x1) {
      return false;
    }
    if (this->y1 != other.y1) {
      return false;
    }
    if (this->x2 != other.x2) {
      return false;
    }
    if (this->y2 != other.y2) {
      return false;
    }
    return true;
  }
  bool operator!=(const Desired_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Desired_

// alias to use template instance with default allocator
using Desired =
  project_interfaces::msg::Desired_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace project_interfaces

#endif  // PROJECT_INTERFACES__MSG__DETAIL__DESIRED__STRUCT_HPP_
