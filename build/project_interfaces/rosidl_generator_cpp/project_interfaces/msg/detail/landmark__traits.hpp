// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from project_interfaces:msg/Landmark.idl
// generated code does not contain a copyright notice

#ifndef PROJECT_INTERFACES__MSG__DETAIL__LANDMARK__TRAITS_HPP_
#define PROJECT_INTERFACES__MSG__DETAIL__LANDMARK__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "project_interfaces/msg/detail/landmark__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace project_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const Landmark & msg,
  std::ostream & out)
{
  out << "{";
  // member: dim
  {
    out << "dim: ";
    rosidl_generator_traits::value_to_yaml(msg.dim, out);
    out << ", ";
  }

  // member: state
  {
    if (msg.state.size() == 0) {
      out << "state: []";
    } else {
      out << "state: [";
      size_t pending_items = msg.state.size();
      for (auto item : msg.state) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: phi
  {
    if (msg.phi.size() == 0) {
      out << "phi: []";
    } else {
      out << "phi: [";
      size_t pending_items = msg.phi.size();
      for (auto item : msg.phi) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: p
  {
    if (msg.p.size() == 0) {
      out << "p: []";
    } else {
      out << "p: [";
      size_t pending_items = msg.p.size();
      for (auto item : msg.p) {
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
  const Landmark & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: dim
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "dim: ";
    rosidl_generator_traits::value_to_yaml(msg.dim, out);
    out << "\n";
  }

  // member: state
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.state.size() == 0) {
      out << "state: []\n";
    } else {
      out << "state:\n";
      for (auto item : msg.state) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: phi
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.phi.size() == 0) {
      out << "phi: []\n";
    } else {
      out << "phi:\n";
      for (auto item : msg.phi) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: p
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.p.size() == 0) {
      out << "p: []\n";
    } else {
      out << "p:\n";
      for (auto item : msg.p) {
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

inline std::string to_yaml(const Landmark & msg, bool use_flow_style = false)
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
  const project_interfaces::msg::Landmark & msg,
  std::ostream & out, size_t indentation = 0)
{
  project_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use project_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const project_interfaces::msg::Landmark & msg)
{
  return project_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<project_interfaces::msg::Landmark>()
{
  return "project_interfaces::msg::Landmark";
}

template<>
inline const char * name<project_interfaces::msg::Landmark>()
{
  return "project_interfaces/msg/Landmark";
}

template<>
struct has_fixed_size<project_interfaces::msg::Landmark>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<project_interfaces::msg::Landmark>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<project_interfaces::msg::Landmark>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // PROJECT_INTERFACES__MSG__DETAIL__LANDMARK__TRAITS_HPP_
