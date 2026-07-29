// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from project_interfaces:msg/MPCprediction.idl
// generated code does not contain a copyright notice

#ifndef PROJECT_INTERFACES__MSG__DETAIL__MP_CPREDICTION__STRUCT_HPP_
#define PROJECT_INTERFACES__MSG__DETAIL__MP_CPREDICTION__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__project_interfaces__msg__MPCprediction __attribute__((deprecated))
#else
# define DEPRECATED__project_interfaces__msg__MPCprediction __declspec(deprecated)
#endif

namespace project_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct MPCprediction_
{
  using Type = MPCprediction_<ContainerAllocator>;

  explicit MPCprediction_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit MPCprediction_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _x_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _x_type x;
  using _y_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _y_type y;
  using _theta_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _theta_type theta;

  // setters for named parameter idiom
  Type & set__x(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->x = _arg;
    return *this;
  }
  Type & set__y(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->y = _arg;
    return *this;
  }
  Type & set__theta(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->theta = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    project_interfaces::msg::MPCprediction_<ContainerAllocator> *;
  using ConstRawPtr =
    const project_interfaces::msg::MPCprediction_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<project_interfaces::msg::MPCprediction_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<project_interfaces::msg::MPCprediction_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      project_interfaces::msg::MPCprediction_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<project_interfaces::msg::MPCprediction_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      project_interfaces::msg::MPCprediction_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<project_interfaces::msg::MPCprediction_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<project_interfaces::msg::MPCprediction_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<project_interfaces::msg::MPCprediction_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__project_interfaces__msg__MPCprediction
    std::shared_ptr<project_interfaces::msg::MPCprediction_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__project_interfaces__msg__MPCprediction
    std::shared_ptr<project_interfaces::msg::MPCprediction_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MPCprediction_ & other) const
  {
    if (this->x != other.x) {
      return false;
    }
    if (this->y != other.y) {
      return false;
    }
    if (this->theta != other.theta) {
      return false;
    }
    return true;
  }
  bool operator!=(const MPCprediction_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MPCprediction_

// alias to use template instance with default allocator
using MPCprediction =
  project_interfaces::msg::MPCprediction_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace project_interfaces

#endif  // PROJECT_INTERFACES__MSG__DETAIL__MP_CPREDICTION__STRUCT_HPP_
