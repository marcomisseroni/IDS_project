#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};


#[link(name = "project_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__project_interfaces__msg__Measurement() -> *const std::ffi::c_void;
}

#[link(name = "project_interfaces__rosidl_generator_c")]
extern "C" {
    fn project_interfaces__msg__Measurement__init(msg: *mut Measurement) -> bool;
    fn project_interfaces__msg__Measurement__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Measurement>, size: usize) -> bool;
    fn project_interfaces__msg__Measurement__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Measurement>);
    fn project_interfaces__msg__Measurement__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Measurement>, out_seq: *mut rosidl_runtime_rs::Sequence<Measurement>) -> bool;
}

// Corresponds to project_interfaces__msg__Measurement
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Measurement {

    // This member is not documented.
    #[allow(missing_docs)]
    pub id_a: i64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub id_b: i64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub x: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub y: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub dtheta: f64,

}



impl Default for Measurement {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !project_interfaces__msg__Measurement__init(&mut msg as *mut _) {
        panic!("Call to project_interfaces__msg__Measurement__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Measurement {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { project_interfaces__msg__Measurement__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { project_interfaces__msg__Measurement__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { project_interfaces__msg__Measurement__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Measurement {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Measurement where Self: Sized {
  const TYPE_NAME: &'static str = "project_interfaces/msg/Measurement";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__project_interfaces__msg__Measurement() }
  }
}


#[link(name = "project_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__project_interfaces__msg__Landmark() -> *const std::ffi::c_void;
}

#[link(name = "project_interfaces__rosidl_generator_c")]
extern "C" {
    fn project_interfaces__msg__Landmark__init(msg: *mut Landmark) -> bool;
    fn project_interfaces__msg__Landmark__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Landmark>, size: usize) -> bool;
    fn project_interfaces__msg__Landmark__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Landmark>);
    fn project_interfaces__msg__Landmark__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Landmark>, out_seq: *mut rosidl_runtime_rs::Sequence<Landmark>) -> bool;
}

// Corresponds to project_interfaces__msg__Landmark
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Landmark {

    // This member is not documented.
    #[allow(missing_docs)]
    pub dim: i32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub id_a: i32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub id_b: i32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub state: rosidl_runtime_rs::Sequence<f64>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub phi: rosidl_runtime_rs::Sequence<f64>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub p: rosidl_runtime_rs::Sequence<f64>,

}



impl Default for Landmark {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !project_interfaces__msg__Landmark__init(&mut msg as *mut _) {
        panic!("Call to project_interfaces__msg__Landmark__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Landmark {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { project_interfaces__msg__Landmark__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { project_interfaces__msg__Landmark__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { project_interfaces__msg__Landmark__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Landmark {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Landmark where Self: Sized {
  const TYPE_NAME: &'static str = "project_interfaces/msg/Landmark";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__project_interfaces__msg__Landmark() }
  }
}


#[link(name = "project_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__project_interfaces__msg__Update() -> *const std::ffi::c_void;
}

#[link(name = "project_interfaces__rosidl_generator_c")]
extern "C" {
    fn project_interfaces__msg__Update__init(msg: *mut Update) -> bool;
    fn project_interfaces__msg__Update__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Update>, size: usize) -> bool;
    fn project_interfaces__msg__Update__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Update>);
    fn project_interfaces__msg__Update__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Update>, out_seq: *mut rosidl_runtime_rs::Sequence<Update>) -> bool;
}

// Corresponds to project_interfaces__msg__Update
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Update {

    // This member is not documented.
    #[allow(missing_docs)]
    pub id_a: i32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub id_b: i32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub dim_a: i32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub dim_b: i32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub ra: rosidl_runtime_rs::Sequence<f64>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub gamma_a: rosidl_runtime_rs::Sequence<f64>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub gamma_b: rosidl_runtime_rs::Sequence<f64>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub w1: rosidl_runtime_rs::Sequence<f64>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub w2: rosidl_runtime_rs::Sequence<f64>,

}



impl Default for Update {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !project_interfaces__msg__Update__init(&mut msg as *mut _) {
        panic!("Call to project_interfaces__msg__Update__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Update {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { project_interfaces__msg__Update__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { project_interfaces__msg__Update__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { project_interfaces__msg__Update__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Update {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Update where Self: Sized {
  const TYPE_NAME: &'static str = "project_interfaces/msg/Update";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__project_interfaces__msg__Update() }
  }
}


#[link(name = "project_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__project_interfaces__msg__State() -> *const std::ffi::c_void;
}

#[link(name = "project_interfaces__rosidl_generator_c")]
extern "C" {
    fn project_interfaces__msg__State__init(msg: *mut State) -> bool;
    fn project_interfaces__msg__State__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<State>, size: usize) -> bool;
    fn project_interfaces__msg__State__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<State>);
    fn project_interfaces__msg__State__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<State>, out_seq: *mut rosidl_runtime_rs::Sequence<State>) -> bool;
}

// Corresponds to project_interfaces__msg__State
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct State {

    // This member is not documented.
    #[allow(missing_docs)]
    pub id: i32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub x: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub y: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub theta: f64,

}



impl Default for State {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !project_interfaces__msg__State__init(&mut msg as *mut _) {
        panic!("Call to project_interfaces__msg__State__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for State {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { project_interfaces__msg__State__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { project_interfaces__msg__State__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { project_interfaces__msg__State__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for State {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for State where Self: Sized {
  const TYPE_NAME: &'static str = "project_interfaces/msg/State";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__project_interfaces__msg__State() }
  }
}


#[link(name = "project_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__project_interfaces__msg__MPCprediction() -> *const std::ffi::c_void;
}

#[link(name = "project_interfaces__rosidl_generator_c")]
extern "C" {
    fn project_interfaces__msg__MPCprediction__init(msg: *mut MPCprediction) -> bool;
    fn project_interfaces__msg__MPCprediction__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<MPCprediction>, size: usize) -> bool;
    fn project_interfaces__msg__MPCprediction__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<MPCprediction>);
    fn project_interfaces__msg__MPCprediction__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<MPCprediction>, out_seq: *mut rosidl_runtime_rs::Sequence<MPCprediction>) -> bool;
}

// Corresponds to project_interfaces__msg__MPCprediction
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct MPCprediction {

    // This member is not documented.
    #[allow(missing_docs)]
    pub x: rosidl_runtime_rs::Sequence<f64>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub y: rosidl_runtime_rs::Sequence<f64>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub theta: rosidl_runtime_rs::Sequence<f64>,

}



impl Default for MPCprediction {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !project_interfaces__msg__MPCprediction__init(&mut msg as *mut _) {
        panic!("Call to project_interfaces__msg__MPCprediction__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for MPCprediction {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { project_interfaces__msg__MPCprediction__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { project_interfaces__msg__MPCprediction__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { project_interfaces__msg__MPCprediction__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for MPCprediction {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for MPCprediction where Self: Sized {
  const TYPE_NAME: &'static str = "project_interfaces/msg/MPCprediction";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__project_interfaces__msg__MPCprediction() }
  }
}


#[link(name = "project_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__project_interfaces__msg__Desired() -> *const std::ffi::c_void;
}

#[link(name = "project_interfaces__rosidl_generator_c")]
extern "C" {
    fn project_interfaces__msg__Desired__init(msg: *mut Desired) -> bool;
    fn project_interfaces__msg__Desired__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Desired>, size: usize) -> bool;
    fn project_interfaces__msg__Desired__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Desired>);
    fn project_interfaces__msg__Desired__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Desired>, out_seq: *mut rosidl_runtime_rs::Sequence<Desired>) -> bool;
}

// Corresponds to project_interfaces__msg__Desired
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Desired {

    // This member is not documented.
    #[allow(missing_docs)]
    pub x0: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub y0: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub x1: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub y1: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub x2: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub y2: f64,

}



impl Default for Desired {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !project_interfaces__msg__Desired__init(&mut msg as *mut _) {
        panic!("Call to project_interfaces__msg__Desired__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Desired {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { project_interfaces__msg__Desired__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { project_interfaces__msg__Desired__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { project_interfaces__msg__Desired__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Desired {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Desired where Self: Sized {
  const TYPE_NAME: &'static str = "project_interfaces/msg/Desired";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__project_interfaces__msg__Desired() }
  }
}


