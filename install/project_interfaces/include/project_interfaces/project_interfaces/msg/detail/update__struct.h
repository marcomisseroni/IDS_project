// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from project_interfaces:msg/Update.idl
// generated code does not contain a copyright notice

#ifndef PROJECT_INTERFACES__MSG__DETAIL__UPDATE__STRUCT_H_
#define PROJECT_INTERFACES__MSG__DETAIL__UPDATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'ra'
// Member 'gamma_a'
// Member 'gamma_b'
// Member 'w1'
// Member 'w2'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/Update in the package project_interfaces.
typedef struct project_interfaces__msg__Update
{
  int32_t id_a;
  int32_t id_b;
  int32_t dim_a;
  int32_t dim_b;
  rosidl_runtime_c__double__Sequence ra;
  rosidl_runtime_c__double__Sequence gamma_a;
  rosidl_runtime_c__double__Sequence gamma_b;
  rosidl_runtime_c__double__Sequence w1;
  rosidl_runtime_c__double__Sequence w2;
} project_interfaces__msg__Update;

// Struct for a sequence of project_interfaces__msg__Update.
typedef struct project_interfaces__msg__Update__Sequence
{
  project_interfaces__msg__Update * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} project_interfaces__msg__Update__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PROJECT_INTERFACES__MSG__DETAIL__UPDATE__STRUCT_H_
