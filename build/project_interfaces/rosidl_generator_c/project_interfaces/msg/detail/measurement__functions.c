// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from project_interfaces:msg/Measurement.idl
// generated code does not contain a copyright notice
#include "project_interfaces/msg/detail/measurement__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
project_interfaces__msg__Measurement__init(project_interfaces__msg__Measurement * msg)
{
  if (!msg) {
    return false;
  }
  // id_a
  // id_b
  // x
  // y
  // dtheta
  return true;
}

void
project_interfaces__msg__Measurement__fini(project_interfaces__msg__Measurement * msg)
{
  if (!msg) {
    return;
  }
  // id_a
  // id_b
  // x
  // y
  // dtheta
}

bool
project_interfaces__msg__Measurement__are_equal(const project_interfaces__msg__Measurement * lhs, const project_interfaces__msg__Measurement * rhs)
{
  if (!lhs || !rhs) {
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
  // x
  if (lhs->x != rhs->x) {
    return false;
  }
  // y
  if (lhs->y != rhs->y) {
    return false;
  }
  // dtheta
  if (lhs->dtheta != rhs->dtheta) {
    return false;
  }
  return true;
}

bool
project_interfaces__msg__Measurement__copy(
  const project_interfaces__msg__Measurement * input,
  project_interfaces__msg__Measurement * output)
{
  if (!input || !output) {
    return false;
  }
  // id_a
  output->id_a = input->id_a;
  // id_b
  output->id_b = input->id_b;
  // x
  output->x = input->x;
  // y
  output->y = input->y;
  // dtheta
  output->dtheta = input->dtheta;
  return true;
}

project_interfaces__msg__Measurement *
project_interfaces__msg__Measurement__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  project_interfaces__msg__Measurement * msg = (project_interfaces__msg__Measurement *)allocator.allocate(sizeof(project_interfaces__msg__Measurement), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(project_interfaces__msg__Measurement));
  bool success = project_interfaces__msg__Measurement__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
project_interfaces__msg__Measurement__destroy(project_interfaces__msg__Measurement * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    project_interfaces__msg__Measurement__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
project_interfaces__msg__Measurement__Sequence__init(project_interfaces__msg__Measurement__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  project_interfaces__msg__Measurement * data = NULL;

  if (size) {
    data = (project_interfaces__msg__Measurement *)allocator.zero_allocate(size, sizeof(project_interfaces__msg__Measurement), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = project_interfaces__msg__Measurement__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        project_interfaces__msg__Measurement__fini(&data[i - 1]);
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
project_interfaces__msg__Measurement__Sequence__fini(project_interfaces__msg__Measurement__Sequence * array)
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
      project_interfaces__msg__Measurement__fini(&array->data[i]);
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

project_interfaces__msg__Measurement__Sequence *
project_interfaces__msg__Measurement__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  project_interfaces__msg__Measurement__Sequence * array = (project_interfaces__msg__Measurement__Sequence *)allocator.allocate(sizeof(project_interfaces__msg__Measurement__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = project_interfaces__msg__Measurement__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
project_interfaces__msg__Measurement__Sequence__destroy(project_interfaces__msg__Measurement__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    project_interfaces__msg__Measurement__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
project_interfaces__msg__Measurement__Sequence__are_equal(const project_interfaces__msg__Measurement__Sequence * lhs, const project_interfaces__msg__Measurement__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!project_interfaces__msg__Measurement__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
project_interfaces__msg__Measurement__Sequence__copy(
  const project_interfaces__msg__Measurement__Sequence * input,
  project_interfaces__msg__Measurement__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(project_interfaces__msg__Measurement);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    project_interfaces__msg__Measurement * data =
      (project_interfaces__msg__Measurement *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!project_interfaces__msg__Measurement__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          project_interfaces__msg__Measurement__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!project_interfaces__msg__Measurement__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
