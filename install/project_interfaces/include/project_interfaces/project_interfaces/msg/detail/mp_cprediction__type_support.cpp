// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from project_interfaces:msg/MPCprediction.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "project_interfaces/msg/detail/mp_cprediction__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace project_interfaces
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void MPCprediction_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) project_interfaces::msg::MPCprediction(_init);
}

void MPCprediction_fini_function(void * message_memory)
{
  auto typed_message = static_cast<project_interfaces::msg::MPCprediction *>(message_memory);
  typed_message->~MPCprediction();
}

size_t size_function__MPCprediction__x(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__MPCprediction__x(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__MPCprediction__x(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__MPCprediction__x(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__MPCprediction__x(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__MPCprediction__x(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__MPCprediction__x(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__MPCprediction__x(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

size_t size_function__MPCprediction__y(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__MPCprediction__y(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__MPCprediction__y(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__MPCprediction__y(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__MPCprediction__y(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__MPCprediction__y(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__MPCprediction__y(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__MPCprediction__y(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

size_t size_function__MPCprediction__theta(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__MPCprediction__theta(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__MPCprediction__theta(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__MPCprediction__theta(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__MPCprediction__theta(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__MPCprediction__theta(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__MPCprediction__theta(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__MPCprediction__theta(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember MPCprediction_message_member_array[3] = {
  {
    "x",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces::msg::MPCprediction, x),  // bytes offset in struct
    nullptr,  // default value
    size_function__MPCprediction__x,  // size() function pointer
    get_const_function__MPCprediction__x,  // get_const(index) function pointer
    get_function__MPCprediction__x,  // get(index) function pointer
    fetch_function__MPCprediction__x,  // fetch(index, &value) function pointer
    assign_function__MPCprediction__x,  // assign(index, value) function pointer
    resize_function__MPCprediction__x  // resize(index) function pointer
  },
  {
    "y",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces::msg::MPCprediction, y),  // bytes offset in struct
    nullptr,  // default value
    size_function__MPCprediction__y,  // size() function pointer
    get_const_function__MPCprediction__y,  // get_const(index) function pointer
    get_function__MPCprediction__y,  // get(index) function pointer
    fetch_function__MPCprediction__y,  // fetch(index, &value) function pointer
    assign_function__MPCprediction__y,  // assign(index, value) function pointer
    resize_function__MPCprediction__y  // resize(index) function pointer
  },
  {
    "theta",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces::msg::MPCprediction, theta),  // bytes offset in struct
    nullptr,  // default value
    size_function__MPCprediction__theta,  // size() function pointer
    get_const_function__MPCprediction__theta,  // get_const(index) function pointer
    get_function__MPCprediction__theta,  // get(index) function pointer
    fetch_function__MPCprediction__theta,  // fetch(index, &value) function pointer
    assign_function__MPCprediction__theta,  // assign(index, value) function pointer
    resize_function__MPCprediction__theta  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers MPCprediction_message_members = {
  "project_interfaces::msg",  // message namespace
  "MPCprediction",  // message name
  3,  // number of fields
  sizeof(project_interfaces::msg::MPCprediction),
  MPCprediction_message_member_array,  // message members
  MPCprediction_init_function,  // function to initialize message memory (memory has to be allocated)
  MPCprediction_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t MPCprediction_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &MPCprediction_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace project_interfaces


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<project_interfaces::msg::MPCprediction>()
{
  return &::project_interfaces::msg::rosidl_typesupport_introspection_cpp::MPCprediction_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, project_interfaces, msg, MPCprediction)() {
  return &::project_interfaces::msg::rosidl_typesupport_introspection_cpp::MPCprediction_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
