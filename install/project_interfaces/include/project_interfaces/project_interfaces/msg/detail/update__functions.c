// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from project_interfaces:msg/Update.idl
// generated code does not contain a copyright notice
#include "project_interfaces/msg/detail/update__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `ra`
// Member `gamma_a`
// Member `gamma_b`
// Member `w1`
// Member `w2`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
project_interfaces__msg__Update__init(project_interfaces__msg__Update * msg)
{
  if (!msg) {
    return false;
  }
  // id_a
  // id_b
  // dim_a
  // dim_b
  // ra
  if (!rosidl_runtime_c__double__Sequence__init(&msg->ra, 0)) {
    project_interfaces__msg__Update__fini(msg);
    return false;
  }
  // gamma_a
  if (!rosidl_runtime_c__double__Sequence__init(&msg->gamma_a, 0)) {
    project_interfaces__msg__Update__fini(msg);
    return false;
  }
  // gamma_b
  if (!rosidl_runtime_c__double__Sequence__init(&msg->gamma_b, 0)) {
    project_interfaces__msg__Update__fini(msg);
    return false;
  }
  // w1
  if (!rosidl_runtime_c__double__Sequence__init(&msg->w1, 0)) {
    project_interfaces__msg__Update__fini(msg);
    return false;
  }
  // w2
  if (!rosidl_runtime_c__double__Sequence__init(&msg->w2, 0)) {
    project_interfaces__msg__Update__fini(msg);
    return false;
  }
  return true;
}

void
project_interfaces__msg__Update__fini(project_interfaces__msg__Update * msg)
{
  if (!msg) {
    return;
  }
  // id_a
  // id_b
  // dim_a
  // dim_b
  // ra
  rosidl_runtime_c__double__Sequence__fini(&msg->ra);
  // gamma_a
  rosidl_runtime_c__double__Sequence__fini(&msg->gamma_a);
  // gamma_b
  rosidl_runtime_c__double__Sequence__fini(&msg->gamma_b);
  // w1
  rosidl_runtime_c__double__Sequence__fini(&msg->w1);
  // w2
  rosidl_runtime_c__double__Sequence__fini(&msg->w2);
}

bool
project_interfaces__msg__Update__are_equal(const project_interfaces__msg__Update * lhs, const project_interfaces__msg__Update * rhs)
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
  // dim_a
  if (lhs->dim_a != rhs->dim_a) {
    return false;
  }
  // dim_b
  if (lhs->dim_b != rhs->dim_b) {
    return false;
  }
  // ra
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->ra), &(rhs->ra)))
  {
    return false;
  }
  // gamma_a
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->gamma_a), &(rhs->gamma_a)))
  {
    return false;
  }
  // gamma_b
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->gamma_b), &(rhs->gamma_b)))
  {
    return false;
  }
  // w1
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->w1), &(rhs->w1)))
  {
    return false;
  }
  // w2
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->w2), &(rhs->w2)))
  {
    return false;
  }
  return true;
}

bool
project_interfaces__msg__Update__copy(
  const project_interfaces__msg__Update * input,
  project_interfaces__msg__Update * output)
{
  if (!input || !output) {
    return false;
  }
  // id_a
  output->id_a = input->id_a;
  // id_b
  output->id_b = input->id_b;
  // dim_a
  output->dim_a = input->dim_a;
  // dim_b
  output->dim_b = input->dim_b;
  // ra
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->ra), &(output->ra)))
  {
    return false;
  }
  // gamma_a
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->gamma_a), &(output->gamma_a)))
  {
    return false;
  }
  // gamma_b
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->gamma_b), &(output->gamma_b)))
  {
    return false;
  }
  // w1
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->w1), &(output->w1)))
  {
    return false;
  }
  // w2
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->w2), &(output->w2)))
  {
    return false;
  }
  return true;
}

project_interfaces__msg__Update *
project_interfaces__msg__Update__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  project_interfaces__msg__Update * msg = (project_interfaces__msg__Update *)allocator.allocate(sizeof(project_interfaces__msg__Update), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(project_interfaces__msg__Update));
  bool success = project_interfaces__msg__Update__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
project_interfaces__msg__Update__destroy(project_interfaces__msg__Update * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    project_interfaces__msg__Update__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
project_interfaces__msg__Update__Sequence__init(project_interfaces__msg__Update__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  project_interfaces__msg__Update * data = NULL;

  if (size) {
    data = (project_interfaces__msg__Update *)allocator.zero_allocate(size, sizeof(project_interfaces__msg__Update), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = project_interfaces__msg__Update__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        project_interfaces__msg__Update__fini(&data[i - 1]);
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
project_interfaces__msg__Update__Sequence__fini(project_interfaces__msg__Update__Sequence * array)
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
      project_interfaces__msg__Update__fini(&array->data[i]);
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

project_interfaces__msg__Update__Sequence *
project_interfaces__msg__Update__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  project_interfaces__msg__Update__Sequence * array = (project_interfaces__msg__Update__Sequence *)allocator.allocate(sizeof(project_interfaces__msg__Update__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = project_interfaces__msg__Update__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
project_interfaces__msg__Update__Sequence__destroy(project_interfaces__msg__Update__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    project_interfaces__msg__Update__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
project_interfaces__msg__Update__Sequence__are_equal(const project_interfaces__msg__Update__Sequence * lhs, const project_interfaces__msg__Update__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!project_interfaces__msg__Update__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
project_interfaces__msg__Update__Sequence__copy(
  const project_interfaces__msg__Update__Sequence * input,
  project_interfaces__msg__Update__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(project_interfaces__msg__Update);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    project_interfaces__msg__Update * data =
      (project_interfaces__msg__Update *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!project_interfaces__msg__Update__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          project_interfaces__msg__Update__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!project_interfaces__msg__Update__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
