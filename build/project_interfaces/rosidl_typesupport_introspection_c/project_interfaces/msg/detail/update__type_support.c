// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from project_interfaces:msg/Update.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "project_interfaces/msg/detail/update__rosidl_typesupport_introspection_c.h"
#include "project_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "project_interfaces/msg/detail/update__functions.h"
#include "project_interfaces/msg/detail/update__struct.h"


// Include directives for member types
// Member `ra`
// Member `gamma_a`
// Member `gamma_b`
// Member `w1`
// Member `w2`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void project_interfaces__msg__Update__rosidl_typesupport_introspection_c__Update_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  project_interfaces__msg__Update__init(message_memory);
}

void project_interfaces__msg__Update__rosidl_typesupport_introspection_c__Update_fini_function(void * message_memory)
{
  project_interfaces__msg__Update__fini(message_memory);
}

size_t project_interfaces__msg__Update__rosidl_typesupport_introspection_c__size_function__Update__ra(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_const_function__Update__ra(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_function__Update__ra(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void project_interfaces__msg__Update__rosidl_typesupport_introspection_c__fetch_function__Update__ra(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_const_function__Update__ra(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void project_interfaces__msg__Update__rosidl_typesupport_introspection_c__assign_function__Update__ra(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_function__Update__ra(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool project_interfaces__msg__Update__rosidl_typesupport_introspection_c__resize_function__Update__ra(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t project_interfaces__msg__Update__rosidl_typesupport_introspection_c__size_function__Update__gamma_a(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_const_function__Update__gamma_a(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_function__Update__gamma_a(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void project_interfaces__msg__Update__rosidl_typesupport_introspection_c__fetch_function__Update__gamma_a(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_const_function__Update__gamma_a(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void project_interfaces__msg__Update__rosidl_typesupport_introspection_c__assign_function__Update__gamma_a(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_function__Update__gamma_a(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool project_interfaces__msg__Update__rosidl_typesupport_introspection_c__resize_function__Update__gamma_a(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t project_interfaces__msg__Update__rosidl_typesupport_introspection_c__size_function__Update__gamma_b(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_const_function__Update__gamma_b(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_function__Update__gamma_b(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void project_interfaces__msg__Update__rosidl_typesupport_introspection_c__fetch_function__Update__gamma_b(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_const_function__Update__gamma_b(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void project_interfaces__msg__Update__rosidl_typesupport_introspection_c__assign_function__Update__gamma_b(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_function__Update__gamma_b(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool project_interfaces__msg__Update__rosidl_typesupport_introspection_c__resize_function__Update__gamma_b(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t project_interfaces__msg__Update__rosidl_typesupport_introspection_c__size_function__Update__w1(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_const_function__Update__w1(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_function__Update__w1(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void project_interfaces__msg__Update__rosidl_typesupport_introspection_c__fetch_function__Update__w1(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_const_function__Update__w1(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void project_interfaces__msg__Update__rosidl_typesupport_introspection_c__assign_function__Update__w1(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_function__Update__w1(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool project_interfaces__msg__Update__rosidl_typesupport_introspection_c__resize_function__Update__w1(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t project_interfaces__msg__Update__rosidl_typesupport_introspection_c__size_function__Update__w2(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_const_function__Update__w2(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_function__Update__w2(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void project_interfaces__msg__Update__rosidl_typesupport_introspection_c__fetch_function__Update__w2(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_const_function__Update__w2(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void project_interfaces__msg__Update__rosidl_typesupport_introspection_c__assign_function__Update__w2(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_function__Update__w2(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool project_interfaces__msg__Update__rosidl_typesupport_introspection_c__resize_function__Update__w2(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember project_interfaces__msg__Update__rosidl_typesupport_introspection_c__Update_message_member_array[9] = {
  {
    "id_a",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces__msg__Update, id_a),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "id_b",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces__msg__Update, id_b),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "dim_a",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces__msg__Update, dim_a),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "dim_b",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces__msg__Update, dim_b),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "ra",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces__msg__Update, ra),  // bytes offset in struct
    NULL,  // default value
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__size_function__Update__ra,  // size() function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_const_function__Update__ra,  // get_const(index) function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_function__Update__ra,  // get(index) function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__fetch_function__Update__ra,  // fetch(index, &value) function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__assign_function__Update__ra,  // assign(index, value) function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__resize_function__Update__ra  // resize(index) function pointer
  },
  {
    "gamma_a",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces__msg__Update, gamma_a),  // bytes offset in struct
    NULL,  // default value
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__size_function__Update__gamma_a,  // size() function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_const_function__Update__gamma_a,  // get_const(index) function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_function__Update__gamma_a,  // get(index) function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__fetch_function__Update__gamma_a,  // fetch(index, &value) function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__assign_function__Update__gamma_a,  // assign(index, value) function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__resize_function__Update__gamma_a  // resize(index) function pointer
  },
  {
    "gamma_b",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces__msg__Update, gamma_b),  // bytes offset in struct
    NULL,  // default value
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__size_function__Update__gamma_b,  // size() function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_const_function__Update__gamma_b,  // get_const(index) function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_function__Update__gamma_b,  // get(index) function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__fetch_function__Update__gamma_b,  // fetch(index, &value) function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__assign_function__Update__gamma_b,  // assign(index, value) function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__resize_function__Update__gamma_b  // resize(index) function pointer
  },
  {
    "w1",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces__msg__Update, w1),  // bytes offset in struct
    NULL,  // default value
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__size_function__Update__w1,  // size() function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_const_function__Update__w1,  // get_const(index) function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_function__Update__w1,  // get(index) function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__fetch_function__Update__w1,  // fetch(index, &value) function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__assign_function__Update__w1,  // assign(index, value) function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__resize_function__Update__w1  // resize(index) function pointer
  },
  {
    "w2",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces__msg__Update, w2),  // bytes offset in struct
    NULL,  // default value
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__size_function__Update__w2,  // size() function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_const_function__Update__w2,  // get_const(index) function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__get_function__Update__w2,  // get(index) function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__fetch_function__Update__w2,  // fetch(index, &value) function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__assign_function__Update__w2,  // assign(index, value) function pointer
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__resize_function__Update__w2  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers project_interfaces__msg__Update__rosidl_typesupport_introspection_c__Update_message_members = {
  "project_interfaces__msg",  // message namespace
  "Update",  // message name
  9,  // number of fields
  sizeof(project_interfaces__msg__Update),
  project_interfaces__msg__Update__rosidl_typesupport_introspection_c__Update_message_member_array,  // message members
  project_interfaces__msg__Update__rosidl_typesupport_introspection_c__Update_init_function,  // function to initialize message memory (memory has to be allocated)
  project_interfaces__msg__Update__rosidl_typesupport_introspection_c__Update_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t project_interfaces__msg__Update__rosidl_typesupport_introspection_c__Update_message_type_support_handle = {
  0,
  &project_interfaces__msg__Update__rosidl_typesupport_introspection_c__Update_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_project_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, project_interfaces, msg, Update)() {
  if (!project_interfaces__msg__Update__rosidl_typesupport_introspection_c__Update_message_type_support_handle.typesupport_identifier) {
    project_interfaces__msg__Update__rosidl_typesupport_introspection_c__Update_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &project_interfaces__msg__Update__rosidl_typesupport_introspection_c__Update_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
