// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from project_interfaces:msg/Landmark.idl
// generated code does not contain a copyright notice
#include "project_interfaces/msg/detail/landmark__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `state`
// Member `phi`
// Member `p`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
project_interfaces__msg__Landmark__init(project_interfaces__msg__Landmark * msg)
{
  if (!msg) {
    return false;
  }
  // dim
  // id_a
  // id_b
  // state
  if (!rosidl_runtime_c__double__Sequence__init(&msg->state, 0)) {
    project_interfaces__msg__Landmark__fini(msg);
    return false;
  }
  // phi
  if (!rosidl_runtime_c__double__Sequence__init(&msg->phi, 0)) {
    project_interfaces__msg__Landmark__fini(msg);
    return false;
  }
  // p
  if (!rosidl_runtime_c__double__Sequence__init(&msg->p, 0)) {
    project_interfaces__msg__Landmark__fini(msg);
    return false;
  }
  return true;
}

void
project_interfaces__msg__Landmark__fini(project_interfaces__msg__Landmark * msg)
{
  if (!msg) {
    return;
  }
  // dim
  // id_a
  // id_b
  // state
  rosidl_runtime_c__double__Sequence__fini(&msg->state);
  // phi
  rosidl_runtime_c__double__Sequence__fini(&msg->phi);
  // p
  rosidl_runtime_c__double__Sequence__fini(&msg->p);
}

bool
project_interfaces__msg__Landmark__are_equal(const project_interfaces__msg__Landmark * lhs, const project_interfaces__msg__Landmark * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // dim
  if (lhs->dim != rhs->dim) {
    return false;
  }
  // id_a
  if (lhs->id_a != rhs->id_a) {
    return false;
  }
  // id_b
  if (lhs->id_b != rhs->id_b) {
    return false;
  }
  // state
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->state), &(rhs->state)))
  {
    return false;
  }
  // phi
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->phi), &(rhs->phi)))
  {
    return false;
  }
  // p
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->p), &(rhs->p)))
  {
    return false;
  }
  return true;
}

bool
project_interfaces__msg__Landmark__copy(
  const project_interfaces__msg__Landmark * input,
  project_interfaces__msg__Landmark * output)
{
  if (!input || !output) {
    return false;
  }
  // dim
  output->dim = input->dim;
  // id_a
  output->id_a = input->id_a;
  // id_b
  output->id_b = input->id_b;
  // state
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->state), &(output->state)))
  {
    return false;
  }
  // phi
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->phi), &(output->phi)))
  {
    return false;
  }
  // p
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->p), &(output->p)))
  {
    return false;
  }
  return true;
}

project_interfaces__msg__Landmark *
project_interfaces__msg__Landmark__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  project_interfaces__msg__Landmark * msg = (project_interfaces__msg__Landmark *)allocator.allocate(sizeof(project_interfaces__msg__Landmark), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(project_interfaces__msg__Landmark));
  bool success = project_interfaces__msg__Landmark__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
project_interfaces__msg__Landmark__destroy(project_interfaces__msg__Landmark * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    project_interfaces__msg__Landmark__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
project_interfaces__msg__Landmark__Sequence__init(project_interfaces__msg__Landmark__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  project_interfaces__msg__Landmark * data = NULL;

  if (size) {
    data = (project_interfaces__msg__Landmark *)allocator.zero_allocate(size, sizeof(project_interfaces__msg__Landmark), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = project_interfaces__msg__Landmark__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        project_interfaces__msg__Landmark__fini(&data[i - 1]);
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
project_interfaces__msg__Landmark__Sequence__fini(project_interfaces__msg__Landmark__Sequence * array)
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
      project_interfaces__msg__Landmark__fini(&array->data[i]);
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

project_interfaces__msg__Landmark__Sequence *
project_interfaces__msg__Landmark__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  project_interfaces__msg__Landmark__Sequence * array = (project_interfaces__msg__Landmark__Sequence *)allocator.allocate(sizeof(project_interfaces__msg__Landmark__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = project_interfaces__msg__Landmark__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
project_interfaces__msg__Landmark__Sequence__destroy(project_interfaces__msg__Landmark__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    project_interfaces__msg__Landmark__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
project_interfaces__msg__Landmark__Sequence__are_equal(const project_interfaces__msg__Landmark__Sequence * lhs, const project_interfaces__msg__Landmark__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!project_interfaces__msg__Landmark__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
project_interfaces__msg__Landmark__Sequence__copy(
  const project_interfaces__msg__Landmark__Sequence * input,
  project_interfaces__msg__Landmark__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(project_interfaces__msg__Landmark);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    project_interfaces__msg__Landmark * data =
      (project_interfaces__msg__Landmark *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!project_interfaces__msg__Landmark__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          project_interfaces__msg__Landmark__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!project_interfaces__msg__Landmark__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
