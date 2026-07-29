// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from project_interfaces:msg/Desired.idl
// generated code does not contain a copyright notice

#ifndef PROJECT_INTERFACES__MSG__DETAIL__DESIRED__TRAITS_HPP_
#define PROJECT_INTERFACES__MSG__DETAIL__DESIRED__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "project_interfaces/msg/detail/desired__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace project_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const Desired & msg,
  std::ostream & out)
{
  out << "{";
  // member: x0
  {
    out << "x0: ";
    rosidl_generator_traits::value_to_yaml(msg.x0, out);
    out << ", ";
  }

  // member: y0
  {
    out << "y0: ";
    rosidl_generator_traits::value_to_yaml(msg.y0, out);
    out << ", ";
  }

  // member: x1
  {
    out << "x1: ";
    rosidl_generator_traits::value_to_yaml(msg.x1, out);
    out << ", ";
  }

  // member: y1
  {
    out << "y1: ";
    rosidl_generator_traits::value_to_yaml(msg.y1, out);
    out << ", ";
  }

  // member: x2
  {
    out << "x2: ";
    rosidl_generator_traits::value_to_yaml(msg.x2, out);
    out << ", ";
  }

  // member: y2
  {
    out << "y2: ";
    rosidl_generator_traits::value_to_yaml(msg.y2, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Desired & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: x0
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "x0: ";
    rosidl_generator_traits::value_to_yaml(msg.x0, out);
    out << "\n";
  }

  // member: y0
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "y0: ";
    rosidl_generator_traits::value_to_yaml(msg.y0, out);
    out << "\n";
  }

  // member: x1
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "x1: ";
    rosidl_generator_traits::value_to_yaml(msg.x1, out);
    out << "\n";
  }

  // member: y1
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "y1: ";
    rosidl_generator_traits::value_to_yaml(msg.y1, out);
    out << "\n";
  }

  // member: x2
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "x2: ";
    rosidl_generator_traits::value_to_yaml(msg.x2, out);
    out << "\n";
  }

  // member: y2
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "y2: ";
    rosidl_generator_traits::value_to_yaml(msg.y2, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Desired & msg, bool use_flow_style = false)
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
  const project_interfaces::msg::Desired & msg,
  std::ostream & out, size_t indentation = 0)
{
  project_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use project_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const project_interfaces::msg::Desired & msg)
{
  return project_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<project_interfaces::msg::Desired>()
{
  return "project_interfaces::msg::Desired";
}

template<>
inline const char * name<project_interfaces::msg::Desired>()
{
  return "project_interfaces/msg/Desired";
}

template<>
struct has_fixed_size<project_interfaces::msg::Desired>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<project_interfaces::msg::Desired>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<project_interfaces::msg::Desired>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // PROJECT_INTERFACES__MSG__DETAIL__DESIRED__TRAITS_HPP_
