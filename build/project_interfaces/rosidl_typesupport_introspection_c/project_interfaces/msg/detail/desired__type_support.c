// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from project_interfaces:msg/Desired.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "project_interfaces/msg/detail/desired__rosidl_typesupport_introspection_c.h"
#include "project_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "project_interfaces/msg/detail/desired__functions.h"
#include "project_interfaces/msg/detail/desired__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void project_interfaces__msg__Desired__rosidl_typesupport_introspection_c__Desired_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  project_interfaces__msg__Desired__init(message_memory);
}

void project_interfaces__msg__Desired__rosidl_typesupport_introspection_c__Desired_fini_function(void * message_memory)
{
  project_interfaces__msg__Desired__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember project_interfaces__msg__Desired__rosidl_typesupport_introspection_c__Desired_message_member_array[6] = {
  {
    "x0",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces__msg__Desired, x0),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "y0",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces__msg__Desired, y0),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "x1",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces__msg__Desired, x1),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "y1",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces__msg__Desired, y1),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "x2",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces__msg__Desired, x2),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "y2",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(project_interfaces__msg__Desired, y2),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers project_interfaces__msg__Desired__rosidl_typesupport_introspection_c__Desired_message_members = {
  "project_interfaces__msg",  // message namespace
  "Desired",  // message name
  6,  // number of fields
  sizeof(project_interfaces__msg__Desired),
  project_interfaces__msg__Desired__rosidl_typesupport_introspection_c__Desired_message_member_array,  // message members
  project_interfaces__msg__Desired__rosidl_typesupport_introspection_c__Desired_init_function,  // function to initialize message memory (memory has to be allocated)
  project_interfaces__msg__Desired__rosidl_typesupport_introspection_c__Desired_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t project_interfaces__msg__Desired__rosidl_typesupport_introspection_c__Desired_message_type_support_handle = {
  0,
  &project_interfaces__msg__Desired__rosidl_typesupport_introspection_c__Desired_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_project_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, project_interfaces, msg, Desired)() {
  if (!project_interfaces__msg__Desired__rosidl_typesupport_introspection_c__Desired_message_type_support_handle.typesupport_identifier) {
    project_interfaces__msg__Desired__rosidl_typesupport_introspection_c__Desired_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &project_interfaces__msg__Desired__rosidl_typesupport_introspection_c__Desired_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
