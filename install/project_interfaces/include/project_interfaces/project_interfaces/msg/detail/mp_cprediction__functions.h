// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from project_interfaces:msg/MPCprediction.idl
// generated code does not contain a copyright notice

#ifndef PROJECT_INTERFACES__MSG__DETAIL__MP_CPREDICTION__FUNCTIONS_H_
#define PROJECT_INTERFACES__MSG__DETAIL__MP_CPREDICTION__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "project_interfaces/msg/rosidl_generator_c__visibility_control.h"

#include "project_interfaces/msg/detail/mp_cprediction__struct.h"

/// Initialize msg/MPCprediction message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * project_interfaces__msg__MPCprediction
 * )) before or use
 * project_interfaces__msg__MPCprediction__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_project_interfaces
bool
project_interfaces__msg__MPCprediction__init(project_interfaces__msg__MPCprediction * msg);

/// Finalize msg/MPCprediction message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_project_interfaces
void
project_interfaces__msg__MPCprediction__fini(project_interfaces__msg__MPCprediction * msg);

/// Create msg/MPCprediction message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * project_interfaces__msg__MPCprediction__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_project_interfaces
project_interfaces__msg__MPCprediction *
project_interfaces__msg__MPCprediction__create();

/// Destroy msg/MPCprediction message.
/**
 * It calls
 * project_interfaces__msg__MPCprediction__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_project_interfaces
void
project_interfaces__msg__MPCprediction__destroy(project_interfaces__msg__MPCprediction * msg);

/// Check for msg/MPCprediction message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_project_interfaces
bool
project_interfaces__msg__MPCprediction__are_equal(const project_interfaces__msg__MPCprediction * lhs, const project_interfaces__msg__MPCprediction * rhs);

/// Copy a msg/MPCprediction message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_project_interfaces
bool
project_interfaces__msg__MPCprediction__copy(
  const project_interfaces__msg__MPCprediction * input,
  project_interfaces__msg__MPCprediction * output);

/// Initialize array of msg/MPCprediction messages.
/**
 * It allocates the memory for the number of elements and calls
 * project_interfaces__msg__MPCprediction__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_project_interfaces
bool
project_interfaces__msg__MPCprediction__Sequence__init(project_interfaces__msg__MPCprediction__Sequence * array, size_t size);

/// Finalize array of msg/MPCprediction messages.
/**
 * It calls
 * project_interfaces__msg__MPCprediction__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_project_interfaces
void
project_interfaces__msg__MPCprediction__Sequence__fini(project_interfaces__msg__MPCprediction__Sequence * array);

/// Create array of msg/MPCprediction messages.
/**
 * It allocates the memory for the array and calls
 * project_interfaces__msg__MPCprediction__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_project_interfaces
project_interfaces__msg__MPCprediction__Sequence *
project_interfaces__msg__MPCprediction__Sequence__create(size_t size);

/// Destroy array of msg/MPCprediction messages.
/**
 * It calls
 * project_interfaces__msg__MPCprediction__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_project_interfaces
void
project_interfaces__msg__MPCprediction__Sequence__destroy(project_interfaces__msg__MPCprediction__Sequence * array);

/// Check for msg/MPCprediction message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_project_interfaces
bool
project_interfaces__msg__MPCprediction__Sequence__are_equal(const project_interfaces__msg__MPCprediction__Sequence * lhs, const project_interfaces__msg__MPCprediction__Sequence * rhs);

/// Copy an array of msg/MPCprediction messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_project_interfaces
bool
project_interfaces__msg__MPCprediction__Sequence__copy(
  const project_interfaces__msg__MPCprediction__Sequence * input,
  project_interfaces__msg__MPCprediction__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // PROJECT_INTERFACES__MSG__DETAIL__MP_CPREDICTION__FUNCTIONS_H_
