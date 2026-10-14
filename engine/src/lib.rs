//! State.io engine: rules, board setup, and the AI built on top of them.

pub mod agent;
pub mod map;
pub mod policy;
pub mod rng;
pub mod roster;
pub mod rules;
pub mod search;
pub mod setup;

pub use map::{Map, MAXN};
pub use rng::Rng;
pub use rules::{Action, Orb, Rules, State, World, NEUTRAL};
