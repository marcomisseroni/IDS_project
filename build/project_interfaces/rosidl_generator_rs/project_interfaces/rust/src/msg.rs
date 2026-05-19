#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};



// Corresponds to project_interfaces__msg__Measurement

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
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
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Measurement::default())
  }
}

impl rosidl_runtime_rs::Message for Measurement {
  type RmwMsg = super::msg::rmw::Measurement;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        id_a: msg.id_a,
        id_b: msg.id_b,
        x: msg.x,
        y: msg.y,
        dtheta: msg.dtheta,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      id_a: msg.id_a,
      id_b: msg.id_b,
      x: msg.x,
      y: msg.y,
      dtheta: msg.dtheta,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      id_a: msg.id_a,
      id_b: msg.id_b,
      x: msg.x,
      y: msg.y,
      dtheta: msg.dtheta,
    }
  }
}


// Corresponds to project_interfaces__msg__Landmark

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Landmark {

    // This member is not documented.
    #[allow(missing_docs)]
    pub dim: i32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub state: Vec<f64>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub phi: Vec<f64>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub p: Vec<f64>,

}



impl Default for Landmark {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Landmark::default())
  }
}

impl rosidl_runtime_rs::Message for Landmark {
  type RmwMsg = super::msg::rmw::Landmark;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        dim: msg.dim,
        state: msg.state.into(),
        phi: msg.phi.into(),
        p: msg.p.into(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      dim: msg.dim,
        state: msg.state.as_slice().into(),
        phi: msg.phi.as_slice().into(),
        p: msg.p.as_slice().into(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      dim: msg.dim,
      state: msg.state
          .into_iter()
          .collect(),
      phi: msg.phi
          .into_iter()
          .collect(),
      p: msg.p
          .into_iter()
          .collect(),
    }
  }
}


// Corresponds to project_interfaces__msg__Update

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
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
    pub ra: Vec<f64>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub gamma_a: Vec<f64>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub gamma_b: Vec<f64>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub w1: Vec<f64>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub w2: Vec<f64>,

}



impl Default for Update {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Update::default())
  }
}

impl rosidl_runtime_rs::Message for Update {
  type RmwMsg = super::msg::rmw::Update;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        id_a: msg.id_a,
        id_b: msg.id_b,
        dim_a: msg.dim_a,
        dim_b: msg.dim_b,
        ra: msg.ra.into(),
        gamma_a: msg.gamma_a.into(),
        gamma_b: msg.gamma_b.into(),
        w1: msg.w1.into(),
        w2: msg.w2.into(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      id_a: msg.id_a,
      id_b: msg.id_b,
      dim_a: msg.dim_a,
      dim_b: msg.dim_b,
        ra: msg.ra.as_slice().into(),
        gamma_a: msg.gamma_a.as_slice().into(),
        gamma_b: msg.gamma_b.as_slice().into(),
        w1: msg.w1.as_slice().into(),
        w2: msg.w2.as_slice().into(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      id_a: msg.id_a,
      id_b: msg.id_b,
      dim_a: msg.dim_a,
      dim_b: msg.dim_b,
      ra: msg.ra
          .into_iter()
          .collect(),
      gamma_a: msg.gamma_a
          .into_iter()
          .collect(),
      gamma_b: msg.gamma_b
          .into_iter()
          .collect(),
      w1: msg.w1
          .into_iter()
          .collect(),
      w2: msg.w2
          .into_iter()
          .collect(),
    }
  }
}


// Corresponds to project_interfaces__msg__State

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
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
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::State::default())
  }
}

impl rosidl_runtime_rs::Message for State {
  type RmwMsg = super::msg::rmw::State;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        id: msg.id,
        x: msg.x,
        y: msg.y,
        theta: msg.theta,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      id: msg.id,
      x: msg.x,
      y: msg.y,
      theta: msg.theta,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      id: msg.id,
      x: msg.x,
      y: msg.y,
      theta: msg.theta,
    }
  }
}


// Corresponds to project_interfaces__msg__MPCprediction

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct MPCprediction {

    // This member is not documented.
    #[allow(missing_docs)]
    pub x: Vec<f64>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub y: Vec<f64>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub theta: Vec<f64>,

}



impl Default for MPCprediction {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::MPCprediction::default())
  }
}

impl rosidl_runtime_rs::Message for MPCprediction {
  type RmwMsg = super::msg::rmw::MPCprediction;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        x: msg.x.into(),
        y: msg.y.into(),
        theta: msg.theta.into(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        x: msg.x.as_slice().into(),
        y: msg.y.as_slice().into(),
        theta: msg.theta.as_slice().into(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      x: msg.x
          .into_iter()
          .collect(),
      y: msg.y
          .into_iter()
          .collect(),
      theta: msg.theta
          .into_iter()
          .collect(),
    }
  }
}


