// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from project_interfaces:msg/MPCprediction.idl
// generated code does not contain a copyright notice

#ifndef PROJECT_INTERFACES__MSG__DETAIL__MP_CPREDICTION__STRUCT_H_
#define PROJECT_INTERFACES__MSG__DETAIL__MP_CPREDICTION__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'x'
// Member 'y'
// Member 'theta'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/MPCprediction in the package project_interfaces.
typedef struct project_interfaces__msg__MPCprediction
{
  rosidl_runtime_c__double__Sequence x;
  rosidl_runtime_c__double__Sequence y;
  rosidl_runtime_c__double__Sequence theta;
} project_interfaces__msg__MPCprediction;

// Struct for a sequence of project_interfaces__msg__MPCprediction.
typedef struct project_interfaces__msg__MPCprediction__Sequence
{
  project_interfaces__msg__MPCprediction * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} project_interfaces__msg__MPCprediction__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PROJECT_INTERFACES__MSG__DETAIL__MP_CPREDICTION__STRUCT_H_
