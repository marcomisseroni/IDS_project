// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from project_interfaces:msg/Landmark.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "project_interfaces/msg/detail/landmark__rosidl_typesupport_introspection_c.h"
#include "project_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "project_interfaces/msg/detail/landmark__functions.h"
#include "project_interfaces/msg/detail/landmark__struct.h"


// Include directives for member types
// Member `state`
// Member `phi`
// Member `p`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__Landmark_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  project_interfaces__msg__Landmark__init(message_memory);
}

void project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__Landmark_fini_function(void * message_memory)
{
  project_interfaces__msg__Landmark__fini(message_memory);
}

size_t project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__size_function__Landmark__state(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__get_const_function__Landmark__state(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__get_function__Landmark__state(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__fetch_function__Landmark__state(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__get_const_function__Landmark__state(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__assign_function__Landmark__state(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__get_function__Landmark__state(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__resize_function__Landmark__state(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__size_function__Landmark__phi(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__get_const_function__Landmark__phi(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__get_function__Landmark__phi(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__fetch_function__Landmark__phi(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__get_const_function__Landmark__phi(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__assign_function__Landmark__phi(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__get_function__Landmark__phi(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__resize_function__Landmark__phi(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__size_function__Landmark__p(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__get_const_function__Landmark__p(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__get_function__Landmark__p(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__fetch_function__Landmark__p(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__get_const_function__Landmark__p(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__assign_function__Landmark__p(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__get_function__Landmark__p(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__resize_function__Landmark__p(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__Landmark_message_member_array[4] = {
  {
    "dim",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces__msg__Landmark, dim),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "state",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces__msg__Landmark, state),  // bytes offset in struct
    NULL,  // default value
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__size_function__Landmark__state,  // size() function pointer
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__get_const_function__Landmark__state,  // get_const(index) function pointer
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__get_function__Landmark__state,  // get(index) function pointer
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__fetch_function__Landmark__state,  // fetch(index, &value) function pointer
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__assign_function__Landmark__state,  // assign(index, value) function pointer
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__resize_function__Landmark__state  // resize(index) function pointer
  },
  {
    "phi",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces__msg__Landmark, phi),  // bytes offset in struct
    NULL,  // default value
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__size_function__Landmark__phi,  // size() function pointer
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__get_const_function__Landmark__phi,  // get_const(index) function pointer
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__get_function__Landmark__phi,  // get(index) function pointer
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__fetch_function__Landmark__phi,  // fetch(index, &value) function pointer
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__assign_function__Landmark__phi,  // assign(index, value) function pointer
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__resize_function__Landmark__phi  // resize(index) function pointer
  },
  {
    "p",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces__msg__Landmark, p),  // bytes offset in struct
    NULL,  // default value
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__size_function__Landmark__p,  // size() function pointer
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__get_const_function__Landmark__p,  // get_const(index) function pointer
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__get_function__Landmark__p,  // get(index) function pointer
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__fetch_function__Landmark__p,  // fetch(index, &value) function pointer
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__assign_function__Landmark__p,  // assign(index, value) function pointer
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__resize_function__Landmark__p  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__Landmark_message_members = {
  "project_interfaces__msg",  // message namespace
  "Landmark",  // message name
  4,  // number of fields
  sizeof(project_interfaces__msg__Landmark),
  project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__Landmark_message_member_array,  // message members
  project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__Landmark_init_function,  // function to initialize message memory (memory has to be allocated)
  project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__Landmark_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__Landmark_message_type_support_handle = {
  0,
  &project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__Landmark_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_project_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, project_interfaces, msg, Landmark)() {
  if (!project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__Landmark_message_type_support_handle.typesupport_identifier) {
    project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__Landmark_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &project_interfaces__msg__Landmark__rosidl_typesupport_introspection_c__Landmark_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
