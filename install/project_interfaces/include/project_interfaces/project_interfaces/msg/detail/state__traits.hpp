// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from project_interfaces:msg/State.idl
// generated code does not contain a copyright notice

#ifndef PROJECT_INTERFACES__MSG__DETAIL__STATE__TRAITS_HPP_
#define PROJECT_INTERFACES__MSG__DETAIL__STATE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "project_interfaces/msg/detail/state__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace project_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const State & msg,
  std::ostream & out)
{
  out << "{";
  // member: id
  {
    out << "id: ";
    rosidl_generator_traits::value_to_yaml(msg.id, out);
    out << ", ";
  }

  // member: x
  {
    out << "x: ";
    rosidl_generator_traits::value_to_yaml(msg.x, out);
    out << ", ";
  }

  // member: y
  {
    out << "y: ";
    rosidl_generator_traits::value_to_yaml(msg.y, out);
    out << ", ";
  }

  // member: theta
  {
    out << "theta: ";
    rosidl_generator_traits::value_to_yaml(msg.theta, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const State & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "id: ";
    rosidl_generator_traits::value_to_yaml(msg.id, out);
    out << "\n";
  }

  // member: x
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "x: ";
    rosidl_generator_traits::value_to_yaml(msg.x, out);
    out << "\n";
  }

  // member: y
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "y: ";
    rosidl_generator_traits::value_to_yaml(msg.y, out);
    out << "\n";
  }

  // member: theta
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "theta: ";
    rosidl_generator_traits::value_to_yaml(msg.theta, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const State & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace project_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use project_interfaces::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const project_interfaces::msg::State & msg,
  std::ostream & out, size_t indentation = 0)
{
  project_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use project_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const project_interfaces::msg::State & msg)
{
  return project_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<project_interfaces::msg::State>()
{
  return "project_interfaces::msg::State";
}

template<>
inline const char * name<project_interfaces::msg::State>()
{
  return "project_interfaces/msg/State";
}

template<>
struct has_fixed_size<project_interfaces::msg::State>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<project_interfaces::msg::State>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<project_interfaces::msg::State>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // PROJECT_INTERFACES__MSG__DETAIL__STATE__TRAITS_HPP_
