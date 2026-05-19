// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from project_interfaces:msg/State.idl
// generated code does not contain a copyright notice
#include "project_interfaces/msg/detail/state__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
project_interfaces__msg__State__init(project_interfaces__msg__State * msg)
{
  if (!msg) {
    return false;
  }
  // id
  // x
  // y
  // theta
  return true;
}

void
project_interfaces__msg__State__fini(project_interfaces__msg__State * msg)
{
  if (!msg) {
    return;
  }
  // id
  // x
  // y
  // theta
}

bool
project_interfaces__msg__State__are_equal(const project_interfaces__msg__State * lhs, const project_interfaces__msg__State * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // id
  if (lhs->id != rhs->id) {
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
  // theta
  if (lhs->theta != rhs->theta) {
    return false;
  }
  return true;
}

bool
project_interfaces__msg__State__copy(
  const project_interfaces__msg__State * input,
  project_interfaces__msg__State * output)
{
  if (!input || !output) {
    return false;
  }
  // id
  output->id = input->id;
  // x
  output->x = input->x;
  // y
  output->y = input->y;
  // theta
  output->theta = input->theta;
  return true;
}

project_interfaces__msg__State *
project_interfaces__msg__State__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  project_interfaces__msg__State * msg = (project_interfaces__msg__State *)allocator.allocate(sizeof(project_interfaces__msg__State), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(project_interfaces__msg__State));
  bool success = project_interfaces__msg__State__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
project_interfaces__msg__State__destroy(project_interfaces__msg__State * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    project_interfaces__msg__State__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
project_interfaces__msg__State__Sequence__init(project_interfaces__msg__State__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  project_interfaces__msg__State * data = NULL;

  if (size) {
    data = (project_interfaces__msg__State *)allocator.zero_allocate(size, sizeof(project_interfaces__msg__State), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = project_interfaces__msg__State__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        project_interfaces__msg__State__fini(&data[i - 1]);
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
project_interfaces__msg__State__Sequence__fini(project_interfaces__msg__State__Sequence * array)
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
      project_interfaces__msg__State__fini(&array->data[i]);
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

project_interfaces__msg__State__Sequence *
project_interfaces__msg__State__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  project_interfaces__msg__State__Sequence * array = (project_interfaces__msg__State__Sequence *)allocator.allocate(sizeof(project_interfaces__msg__State__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = project_interfaces__msg__State__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
project_interfaces__msg__State__Sequence__destroy(project_interfaces__msg__State__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    project_interfaces__msg__State__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
project_interfaces__msg__State__Sequence__are_equal(const project_interfaces__msg__State__Sequence * lhs, const project_interfaces__msg__State__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!project_interfaces__msg__State__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
project_interfaces__msg__State__Sequence__copy(
  const project_interfaces__msg__State__Sequence * input,
  project_interfaces__msg__State__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(project_interfaces__msg__State);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    project_interfaces__msg__State * data =
      (project_interfaces__msg__State *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!project_interfaces__msg__State__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          project_interfaces__msg__State__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!project_interfaces__msg__State__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
