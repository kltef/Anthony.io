//! Named agents: one place that turns a spec string into an agent.

use crate::agent::{Agent, Idle, Scripted};
use crate::policy::PolicyParams;
use crate::search::{SearchParams, Searcher};

/// Tuned policies shipped with the engine: one `name: key=value ...` per line.
pub const POLICIES: &str = include_str!("../data/policies.txt");

/// Look a policy up by name: the hand-written styles first, then `data/policies.txt`.
pub fn policy(name: &str) -> Option<PolicyParams> {
    Some(match name {
        "base" => PolicyParams::default(),
        "rush" => PolicyParams::rush(),
        "econ" => PolicyParams::economist(),
        "farmer" => PolicyParams::farmer(),
        "turtle" => PolicyParams::turtle(),
        _ => {
            let line = POLICIES.lines().find(|l| l.split_once(':').map_or(false, |(n, _)| n.trim() == name))?;
            PolicyParams::from_text(line.split_once(':').unwrap().1)
        }
    })
}

/// Specs: a policy name (`base`, `rush`, `econ`, `farmer`, `turtle`, or one from policies.txt),
/// `idle`, or `search`. An optional `@cadence` suffix sets the seconds between decisions.
pub fn make(spec: &str) -> Box<dyn Agent> {
    let (name, cadence) = match spec.split_once('@') {
        Some((n, c)) => (n, c.parse().expect("bad cadence")),
        None => (spec, 0.5),
    };
    match name {
        "idle" => Box::new(Idle),
        _ if name == "search" || name.starts_with("search:") => {
            let mut p = SearchParams::default();
            if spec.contains('@') {
                p.cadence = cadence;
            }
            // search:key=value,key=value ... overrides (see `SearchParams::set`)
            if let Some((_, opts)) = name.split_once(':') {
                for kv in opts.split(',') {
                    let (k, v) = kv.split_once('=').unwrap_or_else(|| panic!("bad search option {kv}"));
                    p.set(k, v);
                }
            }
            Box::new(Searcher::new(spec, p))
        }
        _ => Box::new(Scripted::new(spec, policy(name).unwrap_or_else(|| panic!("unknown agent spec: {spec}")), cadence)),
    }
}

pub const SCRIPTED: [&str; 5] = ["base", "rush", "econ", "farmer", "turtle"];
