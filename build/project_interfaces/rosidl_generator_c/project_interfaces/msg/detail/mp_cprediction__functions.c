// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from project_interfaces:msg/MPCprediction.idl
// generated code does not contain a copyright notice
#include "project_interfaces/msg/detail/mp_cprediction__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `x`
// Member `y`
// Member `theta`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
project_interfaces__msg__MPCprediction__init(project_interfaces__msg__MPCprediction * msg)
{
  if (!msg) {
    return false;
  }
  // x
  if (!rosidl_runtime_c__double__Sequence__init(&msg->x, 0)) {
    project_interfaces__msg__MPCprediction__fini(msg);
    return false;
  }
  // y
  if (!rosidl_runtime_c__double__Sequence__init(&msg->y, 0)) {
    project_interfaces__msg__MPCprediction__fini(msg);
    return false;
  }
  // theta
  if (!rosidl_runtime_c__double__Sequence__init(&msg->theta, 0)) {
    project_interfaces__msg__MPCprediction__fini(msg);
    return false;
  }
  return true;
}

void
project_interfaces__msg__MPCprediction__fini(project_interfaces__msg__MPCprediction * msg)
{
  if (!msg) {
    return;
  }
  // x
  rosidl_runtime_c__double__Sequence__fini(&msg->x);
  // y
  rosidl_runtime_c__double__Sequence__fini(&msg->y);
  // theta
  rosidl_runtime_c__double__Sequence__fini(&msg->theta);
}

bool
project_interfaces__msg__MPCprediction__are_equal(const project_interfaces__msg__MPCprediction * lhs, const project_interfaces__msg__MPCprediction * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // x
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->x), &(rhs->x)))
  {
    return false;
  }
  // y
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->y), &(rhs->y)))
  {
    return false;
  }
  // theta
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->theta), &(rhs->theta)))
  {
    return false;
  }
  return true;
}

bool
project_interfaces__msg__MPCprediction__copy(
  const project_interfaces__msg__MPCprediction * input,
  project_interfaces__msg__MPCprediction * output)
{
  if (!input || !output) {
    return false;
  }
  // x
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->x), &(output->x)))
  {
    return false;
  }
  // y
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->y), &(output->y)))
  {
    return false;
  }
  // theta
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->theta), &(output->theta)))
  {
    return false;
  }
  return true;
}

project_interfaces__msg__MPCprediction *
project_interfaces__msg__MPCprediction__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  project_interfaces__msg__MPCprediction * msg = (project_interfaces__msg__MPCprediction *)allocator.allocate(sizeof(project_interfaces__msg__MPCprediction), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(project_interfaces__msg__MPCprediction));
  bool success = project_interfaces__msg__MPCprediction__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
project_interfaces__msg__MPCprediction__destroy(project_interfaces__msg__MPCprediction * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    project_interfaces__msg__MPCprediction__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
project_interfaces__msg__MPCprediction__Sequence__init(project_interfaces__msg__MPCprediction__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  project_interfaces__msg__MPCprediction * data = NULL;

  if (size) {
    data = (project_interfaces__msg__MPCprediction *)allocator.zero_allocate(size, sizeof(project_interfaces__msg__MPCprediction), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = project_interfaces__msg__MPCprediction__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        project_interfaces__msg__MPCprediction__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
project_interfaces__msg__MPCprediction__Sequence__fini(project_interfaces__msg__MPCprediction__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      project_interfaces__msg__MPCprediction__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

project_interfaces__msg__MPCprediction__Sequence *
project_interfaces__msg__MPCprediction__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  project_interfaces__msg__MPCprediction__Sequence * array = (project_interfaces__msg__MPCprediction__Sequence *)allocator.allocate(sizeof(project_interfaces__msg__MPCprediction__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = project_interfaces__msg__MPCprediction__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
project_interfaces__msg__MPCprediction__Sequence__destroy(project_interfaces__msg__MPCprediction__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    project_interfaces__msg__MPCprediction__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
project_interfaces__msg__MPCprediction__Sequence__are_equal(const project_interfaces__msg__MPCprediction__Sequence * lhs, const project_interfaces__msg__MPCprediction__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!project_interfaces__msg__MPCprediction__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
project_interfaces__msg__MPCprediction__Sequence__copy(
  const project_interfaces__msg__MPCprediction__Sequence * input,
  project_interfaces__msg__MPCprediction__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(project_interfaces__msg__MPCprediction);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    project_interfaces__msg__MPCprediction * data =
      (project_interfaces__msg__MPCprediction *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!project_interfaces__msg__MPCprediction__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          project_interfaces__msg__MPCprediction__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!project_interfaces__msg__MPCprediction__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
