// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from project_interfaces:msg/Measurement.idl
// generated code does not contain a copyright notice

#ifndef PROJECT_INTERFACES__MSG__DETAIL__MEASUREMENT__STRUCT_HPP_
#define PROJECT_INTERFACES__MSG__DETAIL__MEASUREMENT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__project_interfaces__msg__Measurement __attribute__((deprecated))
#else
# define DEPRECATED__project_interfaces__msg__Measurement __declspec(deprecated)
#endif

namespace project_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Measurement_
{
  using Type = Measurement_<ContainerAllocator>;

  explicit Measurement_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id_a = 0ll;
      this->id_b = 0ll;
      this->x = 0.0;
      this->y = 0.0;
      this->dtheta = 0.0;
    }
  }

  explicit Measurement_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id_a = 0ll;
      this->id_b = 0ll;
      this->x = 0.0;
      this->y = 0.0;
      this->dtheta = 0.0;
    }
  }

  // field types and members
  using _id_a_type =
    int64_t;
  _id_a_type id_a;
  using _id_b_type =
    int64_t;
  _id_b_type id_b;
  using _x_type =
    double;
  _x_type x;
  using _y_type =
    double;
  _y_type y;
  using _dtheta_type =
    double;
  _dtheta_type dtheta;

  // setters for named parameter idiom
  Type & set__id_a(
    const int64_t & _arg)
  {
    this->id_a = _arg;
    return *this;
  }
  Type & set__id_b(
    const int64_t & _arg)
  {
    this->id_b = _arg;
    return *this;
  }
  Type & set__x(
    const double & _arg)
  {
    this->x = _arg;
    return *this;
  }
  Type & set__y(
    const double & _arg)
  {
    this->y = _arg;
    return *this;
  }
  Type & set__dtheta(
    const double & _arg)
  {
    this->dtheta = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    project_interfaces::msg::Measurement_<ContainerAllocator> *;
  using ConstRawPtr =
    const project_interfaces::msg::Measurement_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<project_interfaces::msg::Measurement_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<project_interfaces::msg::Measurement_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      project_interfaces::msg::Measurement_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<project_interfaces::msg::Measurement_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      project_interfaces::msg::Measurement_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<project_interfaces::msg::Measurement_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<project_interfaces::msg::Measurement_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<project_interfaces::msg::Measurement_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__project_interfaces__msg__Measurement
    std::shared_ptr<project_interfaces::msg::Measurement_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__project_interfaces__msg__Measurement
    std::shared_ptr<project_interfaces::msg::Measurement_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Measurement_ & other) const
  {
    if (this->id_a != other.id_a) {
      return false;
    }
    if (this->id_b != other.id_b) {
      return false;
    }
    if (this->x != other.x) {
      return false;
    }
    if (this->y != other.y) {
      return false;
    }
    if (this->dtheta != other.dtheta) {
      return false;
    }
    return true;
  }
  bool operator!=(const Measurement_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Measurement_

// alias to use template instance with default allocator
using Measurement =
  project_interfaces::msg::Measurement_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace project_interfaces

#endif  // PROJECT_INTERFACES__MSG__DETAIL__MEASUREMENT__STRUCT_HPP_
