//! Named agents: one place that turns a spec string into an agent.

use crate::agent::{Agent, Idle, Scripted};
use crate::policy::PolicyParams;
use crate::search::{SearchParams, Searcher};

/// Specs: `base`, `rush`, `econ`, `farmer`, `turtle`, `idle`. An optional `@cadence` suffix
/// (seconds between decisions) applies to the scripted agents, e.g. `base@0.3`.
pub fn make(spec: &str) -> Box<dyn Agent> {
    let (name, cadence) = match spec.split_once('@') {
        Some((n, c)) => (n, c.parse().expect("bad cadence")),
        None => (spec, 0.5),
    };
    let params = match name {
        "base" => PolicyParams::default(),
        "rush" => PolicyParams::rush(),
        "econ" => PolicyParams::economist(),
        "farmer" => PolicyParams::farmer(),
        "turtle" => PolicyParams::turtle(),
        "idle" => return Box::new(Idle),
        "search" => {
            let mut p = SearchParams::default();
            if spec.contains('@') {
                p.cadence = cadence;
            }
            return Box::new(Searcher::new(spec, p));
        }
        _ => panic!("unknown agent spec: {spec}"),
    };
    Box::new(Scripted::new(spec, params, cadence))
}

pub const SCRIPTED: [&str; 5] = ["base", "rush", "econ", "farmer", "turtle"];
