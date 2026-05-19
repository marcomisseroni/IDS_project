// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from project_interfaces:msg/State.idl
// generated code does not contain a copyright notice

#ifndef PROJECT_INTERFACES__MSG__DETAIL__STATE__STRUCT_H_
#define PROJECT_INTERFACES__MSG__DETAIL__STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/State in the package project_interfaces.
typedef struct project_interfaces__msg__State
{
  int32_t id;
  double x;
  double y;
  double theta;
} project_interfaces__msg__State;

// Struct for a sequence of project_interfaces__msg__State.
typedef struct project_interfaces__msg__State__Sequence
{
  project_interfaces__msg__State * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} project_interfaces__msg__State__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PROJECT_INTERFACES__MSG__DETAIL__STATE__STRUCT_H_
