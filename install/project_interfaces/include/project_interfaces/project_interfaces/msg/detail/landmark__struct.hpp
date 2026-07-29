// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from project_interfaces:msg/Landmark.idl
// generated code does not contain a copyright notice

#ifndef PROJECT_INTERFACES__MSG__DETAIL__LANDMARK__STRUCT_HPP_
#define PROJECT_INTERFACES__MSG__DETAIL__LANDMARK__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__project_interfaces__msg__Landmark __attribute__((deprecated))
#else
# define DEPRECATED__project_interfaces__msg__Landmark __declspec(deprecated)
#endif

namespace project_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Landmark_
{
  using Type = Landmark_<ContainerAllocator>;

  explicit Landmark_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->dim = 0l;
      this->id_a = 0l;
      this->id_b = 0l;
    }
  }

  explicit Landmark_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->dim = 0l;
      this->id_a = 0l;
      this->id_b = 0l;
    }
  }

  // field types and members
  using _dim_type =
    int32_t;
  _dim_type dim;
  using _id_a_type =
    int32_t;
  _id_a_type id_a;
  using _id_b_type =
    int32_t;
  _id_b_type id_b;
  using _state_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _state_type state;
  using _phi_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _phi_type phi;
  using _p_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _p_type p;

  // setters for named parameter idiom
  Type & set__dim(
    const int32_t & _arg)
  {
    this->dim = _arg;
    return *this;
  }
  Type & set__id_a(
    const int32_t & _arg)
  {
    this->id_a = _arg;
    return *this;
  }
  Type & set__id_b(
    const int32_t & _arg)
  {
    this->id_b = _arg;
    return *this;
  }
  Type & set__state(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->state = _arg;
    return *this;
  }
  Type & set__phi(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->phi = _arg;
    return *this;
  }
  Type & set__p(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->p = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    project_interfaces::msg::Landmark_<ContainerAllocator> *;
  using ConstRawPtr =
    const project_interfaces::msg::Landmark_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<project_interfaces::msg::Landmark_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<project_interfaces::msg::Landmark_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      project_interfaces::msg::Landmark_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<project_interfaces::msg::Landmark_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      project_interfaces::msg::Landmark_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<project_interfaces::msg::Landmark_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<project_interfaces::msg::Landmark_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<project_interfaces::msg::Landmark_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__project_interfaces__msg__Landmark
    std::shared_ptr<project_interfaces::msg::Landmark_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__project_interfaces__msg__Landmark
    std::shared_ptr<project_interfaces::msg::Landmark_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Landmark_ & other) const
  {
    if (this->dim != other.dim) {
      return false;
    }
    if (this->id_a != other.id_a) {
      return false;
    }
    if (this->id_b != other.id_b) {
      return false;
    }
    if (this->state != other.state) {
      return false;
    }
    if (this->phi != other.phi) {
      return false;
    }
    if (this->p != other.p) {
      return false;
    }
    return true;
  }
  bool operator!=(const Landmark_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Landmark_

// alias to use template instance with default allocator
using Landmark =
  project_interfaces::msg::Landmark_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace project_interfaces

#endif  // PROJECT_INTERFACES__MSG__DETAIL__LANDMARK__STRUCT_HPP_
