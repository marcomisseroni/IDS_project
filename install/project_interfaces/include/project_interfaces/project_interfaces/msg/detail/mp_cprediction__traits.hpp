// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from project_interfaces:msg/MPCprediction.idl
// generated code does not contain a copyright notice

#ifndef PROJECT_INTERFACES__MSG__DETAIL__MP_CPREDICTION__TRAITS_HPP_
#define PROJECT_INTERFACES__MSG__DETAIL__MP_CPREDICTION__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "project_interfaces/msg/detail/mp_cprediction__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace project_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const MPCprediction & msg,
  std::ostream & out)
{
  out << "{";
  // member: x
  {
    if (msg.x.size() == 0) {
      out << "x: []";
    } else {
      out << "x: [";
      size_t pending_items = msg.x.size();
      for (auto item : msg.x) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: y
  {
    if (msg.y.size() == 0) {
      out << "y: []";
    } else {
      out << "y: [";
      size_t pending_items = msg.y.size();
      for (auto item : msg.y) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: theta
  {
    if (msg.theta.size() == 0) {
      out << "theta: []";
    } else {
      out << "theta: [";
      size_t pending_items = msg.theta.size();
      for (auto item : msg.theta) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const MPCprediction & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: x
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.x.size() == 0) {
      out << "x: []\n";
    } else {
      out << "x:\n";
      for (auto item : msg.x) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: y
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.y.size() == 0) {
      out << "y: []\n";
    } else {
      out << "y:\n";
      for (auto item : msg.y) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: theta
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.theta.size() == 0) {
      out << "theta: []\n";
    } else {
      out << "theta:\n";
      for (auto item : msg.theta) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const MPCprediction & msg, bool use_flow_style = false)
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
  const project_interfaces::msg::MPCprediction & msg,
  std::ostream & out, size_t indentation = 0)
{
  project_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use project_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const project_interfaces::msg::MPCprediction & msg)
{
  return project_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<project_interfaces::msg::MPCprediction>()
{
  return "project_interfaces::msg::MPCprediction";
}

template<>
inline const char * name<project_interfaces::msg::MPCprediction>()
{
  return "project_interfaces/msg/MPCprediction";
}

template<>
struct has_fixed_size<project_interfaces::msg::MPCprediction>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<project_interfaces::msg::MPCprediction>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<project_interfaces::msg::MPCprediction>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // PROJECT_INTERFACES__MSG__DETAIL__MP_CPREDICTION__TRAITS_HPP_
