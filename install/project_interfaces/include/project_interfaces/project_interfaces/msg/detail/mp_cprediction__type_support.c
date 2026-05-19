// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from project_interfaces:msg/MPCprediction.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "project_interfaces/msg/detail/mp_cprediction__rosidl_typesupport_introspection_c.h"
#include "project_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "project_interfaces/msg/detail/mp_cprediction__functions.h"
#include "project_interfaces/msg/detail/mp_cprediction__struct.h"


// Include directives for member types
// Member `x`
// Member `y`
// Member `theta`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__MPCprediction_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  project_interfaces__msg__MPCprediction__init(message_memory);
}

void project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__MPCprediction_fini_function(void * message_memory)
{
  project_interfaces__msg__MPCprediction__fini(message_memory);
}

size_t project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__size_function__MPCprediction__x(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__get_const_function__MPCprediction__x(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__get_function__MPCprediction__x(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__fetch_function__MPCprediction__x(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__get_const_function__MPCprediction__x(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__assign_function__MPCprediction__x(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__get_function__MPCprediction__x(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__resize_function__MPCprediction__x(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__size_function__MPCprediction__y(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__get_const_function__MPCprediction__y(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__get_function__MPCprediction__y(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__fetch_function__MPCprediction__y(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__get_const_function__MPCprediction__y(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__assign_function__MPCprediction__y(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__get_function__MPCprediction__y(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__resize_function__MPCprediction__y(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__size_function__MPCprediction__theta(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__get_const_function__MPCprediction__theta(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__get_function__MPCprediction__theta(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__fetch_function__MPCprediction__theta(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__get_const_function__MPCprediction__theta(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__assign_function__MPCprediction__theta(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__get_function__MPCprediction__theta(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__resize_function__MPCprediction__theta(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__MPCprediction_message_member_array[3] = {
  {
    "x",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces__msg__MPCprediction, x),  // bytes offset in struct
    NULL,  // default value
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__size_function__MPCprediction__x,  // size() function pointer
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__get_const_function__MPCprediction__x,  // get_const(index) function pointer
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__get_function__MPCprediction__x,  // get(index) function pointer
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__fetch_function__MPCprediction__x,  // fetch(index, &value) function pointer
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__assign_function__MPCprediction__x,  // assign(index, value) function pointer
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__resize_function__MPCprediction__x  // resize(index) function pointer
  },
  {
    "y",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces__msg__MPCprediction, y),  // bytes offset in struct
    NULL,  // default value
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__size_function__MPCprediction__y,  // size() function pointer
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__get_const_function__MPCprediction__y,  // get_const(index) function pointer
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__get_function__MPCprediction__y,  // get(index) function pointer
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__fetch_function__MPCprediction__y,  // fetch(index, &value) function pointer
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__assign_function__MPCprediction__y,  // assign(index, value) function pointer
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__resize_function__MPCprediction__y  // resize(index) function pointer
  },
  {
    "theta",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces__msg__MPCprediction, theta),  // bytes offset in struct
    NULL,  // default value
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__size_function__MPCprediction__theta,  // size() function pointer
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__get_const_function__MPCprediction__theta,  // get_const(index) function pointer
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__get_function__MPCprediction__theta,  // get(index) function pointer
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__fetch_function__MPCprediction__theta,  // fetch(index, &value) function pointer
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__assign_function__MPCprediction__theta,  // assign(index, value) function pointer
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__resize_function__MPCprediction__theta  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__MPCprediction_message_members = {
  "project_interfaces__msg",  // message namespace
  "MPCprediction",  // message name
  3,  // number of fields
  sizeof(project_interfaces__msg__MPCprediction),
  project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__MPCprediction_message_member_array,  // message members
  project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__MPCprediction_init_function,  // function to initialize message memory (memory has to be allocated)
  project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__MPCprediction_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__MPCprediction_message_type_support_handle = {
  0,
  &project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__MPCprediction_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_project_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, project_interfaces, msg, MPCprediction)() {
  if (!project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__MPCprediction_message_type_support_handle.typesupport_identifier) {
    project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__MPCprediction_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &project_interfaces__msg__MPCprediction__rosidl_typesupport_introspection_c__MPCprediction_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
