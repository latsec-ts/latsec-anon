//! Research prototype of SEC secret sharing and the LaTSEC threshold signature.
//!
//! The algorithms are those of Figure 1 and Appendix I of the full version of the paper; the README of
//! the implementation lists the modules.  Nothing here is hardened or audited.

pub mod builder;
pub mod codec;
pub mod error;
pub mod expander;
pub mod fixture;
pub mod intmat;
pub mod kat;
pub mod latsec;
pub mod params;
pub mod profile;
pub mod published;
pub mod ring;
pub mod sampling;
pub mod scheme;
pub mod sim;
pub mod stats;
pub mod xof;

pub use error::{Error, Result};
