#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};


#[link(name = "theora_image_transport__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__theora_image_transport__msg__Packet() -> *const std::ffi::c_void;
}

#[link(name = "theora_image_transport__rosidl_generator_c")]
extern "C" {
    fn theora_image_transport__msg__Packet__init(msg: *mut Packet) -> bool;
    fn theora_image_transport__msg__Packet__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Packet>, size: usize) -> bool;
    fn theora_image_transport__msg__Packet__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Packet>);
    fn theora_image_transport__msg__Packet__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Packet>, out_seq: *mut rosidl_runtime_rs::Sequence<Packet>) -> bool;
}

// Corresponds to theora_image_transport__msg__Packet
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// ROS message adaptation of the ogg_packet struct from libogg,
/// see http://www.xiph.org/ogg/doc/libogg/ogg_packet.html.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Packet {
    /// Original sensor_msgs/Image header
    pub header: std_msgs::msg::rmw::Header,

    /// Raw Theora packet data (combines packet and bytes fields from ogg_packet)
    pub data: rosidl_runtime_rs::Sequence<u8>,

    /// Flag indicating whether this packet begins a logical bitstream
    pub b_o_s: i32,

    /// Flag indicating whether this packet ends a bitstream
    pub e_o_s: i32,

    /// A number indicating the position of this packet in the decoded data
    pub granulepos: i64,

    /// Sequential number of this packet in the ogg bitstream
    pub packetno: i64,

}



impl Default for Packet {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !theora_image_transport__msg__Packet__init(&mut msg as *mut _) {
        panic!("Call to theora_image_transport__msg__Packet__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Packet {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { theora_image_transport__msg__Packet__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { theora_image_transport__msg__Packet__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { theora_image_transport__msg__Packet__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Packet {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Packet where Self: Sized {
  const TYPE_NAME: &'static str = "theora_image_transport/msg/Packet";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__theora_image_transport__msg__Packet() }
  }
}


