#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};



// Corresponds to theora_image_transport__msg__Packet
/// ROS message adaptation of the ogg_packet struct from libogg,
/// see http://www.xiph.org/ogg/doc/libogg/ogg_packet.html.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Packet {
    /// Original sensor_msgs/Image header
    pub header: std_msgs::msg::Header,

    /// Raw Theora packet data (combines packet and bytes fields from ogg_packet)
    pub data: Vec<u8>,

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
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Packet::default())
  }
}

impl rosidl_runtime_rs::Message for Packet {
  type RmwMsg = super::msg::rmw::Packet;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        data: msg.data.into(),
        b_o_s: msg.b_o_s,
        e_o_s: msg.e_o_s,
        granulepos: msg.granulepos,
        packetno: msg.packetno,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        data: msg.data.as_slice().into(),
      b_o_s: msg.b_o_s,
      e_o_s: msg.e_o_s,
      granulepos: msg.granulepos,
      packetno: msg.packetno,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      data: msg.data
          .into_iter()
          .collect(),
      b_o_s: msg.b_o_s,
      e_o_s: msg.e_o_s,
      granulepos: msg.granulepos,
      packetno: msg.packetno,
    }
  }
}


