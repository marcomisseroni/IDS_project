// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from project_interfaces:msg/Measurement.idl
// generated code does not contain a copyright notice

#ifndef PROJECT_INTERFACES__MSG__DETAIL__MEASUREMENT__STRUCT_H_
#define PROJECT_INTERFACES__MSG__DETAIL__MEASUREMENT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/Measurement in the package project_interfaces.
typedef struct project_interfaces__msg__Measurement
{
  int64_t id_a;
  int64_t id_b;
  double x;
  double y;
  double dtheta;
} project_interfaces__msg__Measurement;

// Struct for a sequence of project_interfaces__msg__Measurement.
typedef struct project_interfaces__msg__Measurement__Sequence
{
  project_interfaces__msg__Measurement * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} project_interfaces__msg__Measurement__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PROJECT_INTERFACES__MSG__DETAIL__MEASUREMENT__STRUCT_H_
