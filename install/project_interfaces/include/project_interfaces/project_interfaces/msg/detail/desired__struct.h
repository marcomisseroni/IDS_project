// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from project_interfaces:msg/Desired.idl
// generated code does not contain a copyright notice

#ifndef PROJECT_INTERFACES__MSG__DETAIL__DESIRED__STRUCT_H_
#define PROJECT_INTERFACES__MSG__DETAIL__DESIRED__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/Desired in the package project_interfaces.
typedef struct project_interfaces__msg__Desired
{
  double x0;
  double y0;
  double x1;
  double y1;
  double x2;
  double y2;
} project_interfaces__msg__Desired;

// Struct for a sequence of project_interfaces__msg__Desired.
typedef struct project_interfaces__msg__Desired__Sequence
{
  project_interfaces__msg__Desired * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} project_interfaces__msg__Desired__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PROJECT_INTERFACES__MSG__DETAIL__DESIRED__STRUCT_H_
