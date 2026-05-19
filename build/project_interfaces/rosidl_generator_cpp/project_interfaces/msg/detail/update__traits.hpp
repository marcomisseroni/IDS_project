// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from project_interfaces:msg/Update.idl
// generated code does not contain a copyright notice

#ifndef PROJECT_INTERFACES__MSG__DETAIL__UPDATE__TRAITS_HPP_
#define PROJECT_INTERFACES__MSG__DETAIL__UPDATE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "project_interfaces/msg/detail/update__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace project_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const Update & msg,
  std::ostream & out)
{
  out << "{";
  // member: id_a
  {
    out << "id_a: ";
    rosidl_generator_traits::value_to_yaml(msg.id_a, out);
    out << ", ";
  }

  // member: id_b
  {
    out << "id_b: ";
    rosidl_generator_traits::value_to_yaml(msg.id_b, out);
    out << ", ";
  }

  // member: dim_a
  {
    out << "dim_a: ";
    rosidl_generator_traits::value_to_yaml(msg.dim_a, out);
    out << ", ";
  }

  // member: dim_b
  {
    out << "dim_b: ";
    rosidl_generator_traits::value_to_yaml(msg.dim_b, out);
    out << ", ";
  }

  // member: ra
  {
    if (msg.ra.size() == 0) {
      out << "ra: []";
    } else {
      out << "ra: [";
      size_t pending_items = msg.ra.size();
      for (auto item : msg.ra) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: gamma_a
  {
    if (msg.gamma_a.size() == 0) {
      out << "gamma_a: []";
    } else {
      out << "gamma_a: [";
      size_t pending_items = msg.gamma_a.size();
      for (auto item : msg.gamma_a) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: gamma_b
  {
    if (msg.gamma_b.size() == 0) {
      out << "gamma_b: []";
    } else {
      out << "gamma_b: [";
      size_t pending_items = msg.gamma_b.size();
      for (auto item : msg.gamma_b) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: w1
  {
    if (msg.w1.size() == 0) {
      out << "w1: []";
    } else {
      out << "w1: [";
      size_t pending_items = msg.w1.size();
      for (auto item : msg.w1) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: w2
  {
    if (msg.w2.size() == 0) {
      out << "w2: []";
    } else {
      out << "w2: [";
      size_t pending_items = msg.w2.size();
      for (auto item : msg.w2) {
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
  const Update & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: id_a
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "id_a: ";
    rosidl_generator_traits::value_to_yaml(msg.id_a, out);
    out << "\n";
  }

  // member: id_b
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "id_b: ";
    rosidl_generator_traits::value_to_yaml(msg.id_b, out);
    out << "\n";
  }

  // member: dim_a
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "dim_a: ";
    rosidl_generator_traits::value_to_yaml(msg.dim_a, out);
    out << "\n";
  }

  // member: dim_b
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "dim_b: ";
    rosidl_generator_traits::value_to_yaml(msg.dim_b, out);
    out << "\n";
  }

  // member: ra
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.ra.size() == 0) {
      out << "ra: []\n";
    } else {
      out << "ra:\n";
      for (auto item : msg.ra) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: gamma_a
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.gamma_a.size() == 0) {
      out << "gamma_a: []\n";
    } else {
      out << "gamma_a:\n";
      for (auto item : msg.gamma_a) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: gamma_b
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.gamma_b.size() == 0) {
      out << "gamma_b: []\n";
    } else {
      out << "gamma_b:\n";
      for (auto item : msg.gamma_b) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: w1
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.w1.size() == 0) {
      out << "w1: []\n";
    } else {
      out << "w1:\n";
      for (auto item : msg.w1) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: w2
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.w2.size() == 0) {
      out << "w2: []\n";
    } else {
      out << "w2:\n";
      for (auto item : msg.w2) {
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

inline std::string to_yaml(const Update & msg, bool use_flow_style = false)
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
  const project_interfaces::msg::Update & msg,
  std::ostream & out, size_t indentation = 0)
{
  project_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use project_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const project_interfaces::msg::Update & msg)
{
  return project_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<project_interfaces::msg::Update>()
{
  return "project_interfaces::msg::Update";
}

template<>
inline const char * name<project_interfaces::msg::Update>()
{
  return "project_interfaces/msg/Update";
}

template<>
struct has_fixed_size<project_interfaces::msg::Update>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<project_interfaces::msg::Update>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<project_interfaces::msg::Update>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // PROJECT_INTERFACES__MSG__DETAIL__UPDATE__TRAITS_HPP_
