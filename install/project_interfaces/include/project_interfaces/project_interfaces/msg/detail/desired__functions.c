// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from project_interfaces:msg/Desired.idl
// generated code does not contain a copyright notice
#include "project_interfaces/msg/detail/desired__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
project_interfaces__msg__Desired__init(project_interfaces__msg__Desired * msg)
{
  if (!msg) {
    return false;
  }
  // x0
  // y0
  // x1
  // y1
  // x2
  // y2
  return true;
}

void
project_interfaces__msg__Desired__fini(project_interfaces__msg__Desired * msg)
{
  if (!msg) {
    return;
  }
  // x0
  // y0
  // x1
  // y1
  // x2
  // y2
}

bool
project_interfaces__msg__Desired__are_equal(const project_interfaces__msg__Desired * lhs, const project_interfaces__msg__Desired * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // x0
  if (lhs->x0 != rhs->x0) {
    return false;
  }
  // y0
  if (lhs->y0 != rhs->y0) {
    return false;
  }
  // x1
  if (lhs->x1 != rhs->x1) {
    return false;
  }
  // y1
  if (lhs->y1 != rhs->y1) {
    return false;
  }
  // x2
  if (lhs->x2 != rhs->x2) {
    return false;
  }
  // y2
  if (lhs->y2 != rhs->y2) {
    return false;
  }
  return true;
}

bool
project_interfaces__msg__Desired__copy(
  const project_interfaces__msg__Desired * input,
  project_interfaces__msg__Desired * output)
{
  if (!input || !output) {
    return false;
  }
  // x0
  output->x0 = input->x0;
  // y0
  output->y0 = input->y0;
  // x1
  output->x1 = input->x1;
  // y1
  output->y1 = input->y1;
  // x2
  output->x2 = input->x2;
  // y2
  output->y2 = input->y2;
  return true;
}

project_interfaces__msg__Desired *
project_interfaces__msg__Desired__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  project_interfaces__msg__Desired * msg = (project_interfaces__msg__Desired *)allocator.allocate(sizeof(project_interfaces__msg__Desired), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(project_interfaces__msg__Desired));
  bool success = project_interfaces__msg__Desired__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
project_interfaces__msg__Desired__destroy(project_interfaces__msg__Desired * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    project_interfaces__msg__Desired__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
project_interfaces__msg__Desired__Sequence__init(project_interfaces__msg__Desired__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  project_interfaces__msg__Desired * data = NULL;

  if (size) {
    data = (project_interfaces__msg__Desired *)allocator.zero_allocate(size, sizeof(project_interfaces__msg__Desired), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = project_interfaces__msg__Desired__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        project_interfaces__msg__Desired__fini(&data[i - 1]);
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
project_interfaces__msg__Desired__Sequence__fini(project_interfaces__msg__Desired__Sequence * array)
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
      project_interfaces__msg__Desired__fini(&array->data[i]);
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

project_interfaces__msg__Desired__Sequence *
project_interfaces__msg__Desired__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  project_interfaces__msg__Desired__Sequence * array = (project_interfaces__msg__Desired__Sequence *)allocator.allocate(sizeof(project_interfaces__msg__Desired__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = project_interfaces__msg__Desired__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
project_interfaces__msg__Desired__Sequence__destroy(project_interfaces__msg__Desired__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    project_interfaces__msg__Desired__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
project_interfaces__msg__Desired__Sequence__are_equal(const project_interfaces__msg__Desired__Sequence * lhs, const project_interfaces__msg__Desired__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!project_interfaces__msg__Desired__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
project_interfaces__msg__Desired__Sequence__copy(
  const project_interfaces__msg__Desired__Sequence * input,
  project_interfaces__msg__Desired__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(project_interfaces__msg__Desired);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    project_interfaces__msg__Desired * data =
      (project_interfaces__msg__Desired *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!project_interfaces__msg__Desired__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          project_interfaces__msg__Desired__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!project_interfaces__msg__Desired__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
