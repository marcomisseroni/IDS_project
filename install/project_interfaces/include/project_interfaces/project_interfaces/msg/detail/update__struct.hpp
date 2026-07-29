// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from project_interfaces:msg/Update.idl
// generated code does not contain a copyright notice

#ifndef PROJECT_INTERFACES__MSG__DETAIL__UPDATE__STRUCT_HPP_
#define PROJECT_INTERFACES__MSG__DETAIL__UPDATE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__project_interfaces__msg__Update __attribute__((deprecated))
#else
# define DEPRECATED__project_interfaces__msg__Update __declspec(deprecated)
#endif

namespace project_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Update_
{
  using Type = Update_<ContainerAllocator>;

  explicit Update_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id_a = 0l;
      this->id_b = 0l;
      this->dim_a = 0l;
      this->dim_b = 0l;
    }
  }

  explicit Update_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id_a = 0l;
      this->id_b = 0l;
      this->dim_a = 0l;
      this->dim_b = 0l;
    }
  }

  // field types and members
  using _id_a_type =
    int32_t;
  _id_a_type id_a;
  using _id_b_type =
    int32_t;
  _id_b_type id_b;
  using _dim_a_type =
    int32_t;
  _dim_a_type dim_a;
  using _dim_b_type =
    int32_t;
  _dim_b_type dim_b;
  using _ra_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _ra_type ra;
  using _gamma_a_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _gamma_a_type gamma_a;
  using _gamma_b_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _gamma_b_type gamma_b;
  using _w1_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _w1_type w1;
  using _w2_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _w2_type w2;

  // setters for named parameter idiom
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
  Type & set__dim_a(
    const int32_t & _arg)
  {
    this->dim_a = _arg;
    return *this;
  }
  Type & set__dim_b(
    const int32_t & _arg)
  {
    this->dim_b = _arg;
    return *this;
  }
  Type & set__ra(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->ra = _arg;
    return *this;
  }
  Type & set__gamma_a(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->gamma_a = _arg;
    return *this;
  }
  Type & set__gamma_b(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->gamma_b = _arg;
    return *this;
  }
  Type & set__w1(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->w1 = _arg;
    return *this;
  }
  Type & set__w2(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->w2 = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    project_interfaces::msg::Update_<ContainerAllocator> *;
  using ConstRawPtr =
    const project_interfaces::msg::Update_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<project_interfaces::msg::Update_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<project_interfaces::msg::Update_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      project_interfaces::msg::Update_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<project_interfaces::msg::Update_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      project_interfaces::msg::Update_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<project_interfaces::msg::Update_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<project_interfaces::msg::Update_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<project_interfaces::msg::Update_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__project_interfaces__msg__Update
    std::shared_ptr<project_interfaces::msg::Update_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__project_interfaces__msg__Update
    std::shared_ptr<project_interfaces::msg::Update_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Update_ & other) const
  {
    if (this->id_a != other.id_a) {
      return false;
    }
    if (this->id_b != other.id_b) {
      return false;
    }
    if (this->dim_a != other.dim_a) {
      return false;
    }
    if (this->dim_b != other.dim_b) {
      return false;
    }
    if (this->ra != other.ra) {
      return false;
    }
    if (this->gamma_a != other.gamma_a) {
      return false;
    }
    if (this->gamma_b != other.gamma_b) {
      return false;
    }
    if (this->w1 != other.w1) {
      return false;
    }
    if (this->w2 != other.w2) {
      return false;
    }
    return true;
  }
  bool operator!=(const Update_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Update_

// alias to use template instance with default allocator
using Update =
  project_interfaces::msg::Update_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace project_interfaces

#endif  // PROJECT_INTERFACES__MSG__DETAIL__UPDATE__STRUCT_HPP_
