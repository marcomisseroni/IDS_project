// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from project_interfaces:msg/Landmark.idl
// generated code does not contain a copyright notice
#include "project_interfaces/msg/detail/landmark__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "project_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "project_interfaces/msg/detail/landmark__struct.h"
#include "project_interfaces/msg/detail/landmark__functions.h"
#include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif

#include "rosidl_runtime_c/primitives_sequence.h"  // p, phi, state
#include "rosidl_runtime_c/primitives_sequence_functions.h"  // p, phi, state

// forward declare type support functions


using _Landmark__ros_msg_type = project_interfaces__msg__Landmark;

static bool _Landmark__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _Landmark__ros_msg_type * ros_message = static_cast<const _Landmark__ros_msg_type *>(untyped_ros_message);
  // Field name: dim
  {
    cdr << ros_message->dim;
  }

  // Field name: id_a
  {
    cdr << ros_message->id_a;
  }

  // Field name: id_b
  {
    cdr << ros_message->id_b;
  }

  // Field name: state
  {
    size_t size = ros_message->state.size;
    auto array_ptr = ros_message->state.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: phi
  {
    size_t size = ros_message->phi.size;
    auto array_ptr = ros_message->phi.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: p
  {
    size_t size = ros_message->p.size;
    auto array_ptr = ros_message->p.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  return true;
}

static bool _Landmark__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _Landmark__ros_msg_type * ros_message = static_cast<_Landmark__ros_msg_type *>(untyped_ros_message);
  // Field name: dim
  {
    cdr >> ros_message->dim;
  }

  // Field name: id_a
  {
    cdr >> ros_message->id_a;
  }

  // Field name: id_b
  {
    cdr >> ros_message->id_b;
  }

  // Field name: state
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);

    // Check there are at least 'size' remaining bytes in the CDR stream before resizing
    auto old_state = cdr.getState();
    bool correct_size = cdr.jump(size);
    cdr.setState(old_state);
    if (!correct_size) {
      fprintf(stderr, "sequence size exceeds remaining buffer\n");
      return false;
    }

    if (ros_message->state.data) {
      rosidl_runtime_c__double__Sequence__fini(&ros_message->state);
    }
    if (!rosidl_runtime_c__double__Sequence__init(&ros_message->state, size)) {
      fprintf(stderr, "failed to create array for field 'state'");
      return false;
    }
    auto array_ptr = ros_message->state.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: phi
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);

    // Check there are at least 'size' remaining bytes in the CDR stream before resizing
    auto old_state = cdr.getState();
    bool correct_size = cdr.jump(size);
    cdr.setState(old_state);
    if (!correct_size) {
      fprintf(stderr, "sequence size exceeds remaining buffer\n");
      return false;
    }

    if (ros_message->phi.data) {
      rosidl_runtime_c__double__Sequence__fini(&ros_message->phi);
    }
    if (!rosidl_runtime_c__double__Sequence__init(&ros_message->phi, size)) {
      fprintf(stderr, "failed to create array for field 'phi'");
      return false;
    }
    auto array_ptr = ros_message->phi.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: p
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);

    // Check there are at least 'size' remaining bytes in the CDR stream before resizing
    auto old_state = cdr.getState();
    bool correct_size = cdr.jump(size);
    cdr.setState(old_state);
    if (!correct_size) {
      fprintf(stderr, "sequence size exceeds remaining buffer\n");
      return false;
    }

    if (ros_message->p.data) {
      rosidl_runtime_c__double__Sequence__fini(&ros_message->p);
    }
    if (!rosidl_runtime_c__double__Sequence__init(&ros_message->p, size)) {
      fprintf(stderr, "failed to create array for field 'p'");
      return false;
    }
    auto array_ptr = ros_message->p.data;
    cdr.deserializeArray(array_ptr, size);
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_project_interfaces
size_t get_serialized_size_project_interfaces__msg__Landmark(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _Landmark__ros_msg_type * ros_message = static_cast<const _Landmark__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name dim
  {
    size_t item_size = sizeof(ros_message->dim);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name id_a
  {
    size_t item_size = sizeof(ros_message->id_a);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name id_b
  {
    size_t item_size = sizeof(ros_message->id_b);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name state
  {
    size_t array_size = ros_message->state.size;
    auto array_ptr = ros_message->state.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name phi
  {
    size_t array_size = ros_message->phi.size;
    auto array_ptr = ros_message->phi.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name p
  {
    size_t array_size = ros_message->p.size;
    auto array_ptr = ros_message->p.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _Landmark__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_project_interfaces__msg__Landmark(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_project_interfaces
size_t max_serialized_size_project_interfaces__msg__Landmark(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;

  // member: dim
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: id_a
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: id_b
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: state
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: phi
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: p
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = project_interfaces__msg__Landmark;
    is_plain =
      (
      offsetof(DataType, p) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _Landmark__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_project_interfaces__msg__Landmark(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_Landmark = {
  "project_interfaces::msg",
  "Landmark",
  _Landmark__cdr_serialize,
  _Landmark__cdr_deserialize,
  _Landmark__get_serialized_size,
  _Landmark__max_serialized_size
};

static rosidl_message_type_support_t _Landmark__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_Landmark,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, project_interfaces, msg, Landmark)() {
  return &_Landmark__type_support;
}

#if defined(__cplusplus)
}
#endif
