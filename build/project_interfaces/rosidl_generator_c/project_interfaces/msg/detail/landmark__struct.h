// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from project_interfaces:msg/Landmark.idl
// generated code does not contain a copyright notice

#ifndef PROJECT_INTERFACES__MSG__DETAIL__LANDMARK__STRUCT_H_
#define PROJECT_INTERFACES__MSG__DETAIL__LANDMARK__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'state'
// Member 'phi'
// Member 'p'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/Landmark in the package project_interfaces.
typedef struct project_interfaces__msg__Landmark
{
  int32_t dim;
  int32_t id_a;
  int32_t id_b;
  rosidl_runtime_c__double__Sequence state;
  rosidl_runtime_c__double__Sequence phi;
  rosidl_runtime_c__double__Sequence p;
} project_interfaces__msg__Landmark;

// Struct for a sequence of project_interfaces__msg__Landmark.
typedef struct project_interfaces__msg__Landmark__Sequence
{
  project_interfaces__msg__Landmark * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} project_interfaces__msg__Landmark__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PROJECT_INTERFACES__MSG__DETAIL__LANDMARK__STRUCT_H_
