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
        _ if name == "chris" || name.starts_with("chris:") => {
            let mut p = crate::chris::ChrisParams::default();
            if let Some((_, opts)) = name.split_once(':') {
                for kv in opts.split(',').filter(|x| !x.is_empty()) {
                    let (k, v) = kv.split_once('=').unwrap_or_else(|| panic!("bad chris option {kv}"));
                    p.set(k, v).unwrap_or_else(|e| panic!("{e}"));
                }
            }
            if spec.contains('@') {
                p.cadence = cadence;
            }
            Box::new(crate::chris::Chris::new(spec, p))
        }
        _ if name == "plan" || name.starts_with("plan:") => Box::new(make_planner(spec).unwrap_or_else(|e| panic!("{e}"))),
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

/// Named planner settings, one per difficulty tier. A spec `plan:<tier>` uses these; any further
/// `,key=value` pairs override them (see `PlanParams::set`).
pub fn plan_tier(name: &str) -> Option<&'static str> {
    Some(match name {
        "easy" => "cadence=2.6,h=10,screen=1,samples=2,k=3,cands=24,waits=0/4,feeders=2,refine=0,pess=0,temp=0.03,port=base,pool=base+rush,self=base",
        "medium" => "cadence=1.4,h=12,screen=1,samples=2,k=4,cands=40,waits=0/4,feeders=2,refine=0,pess=0.25,temp=0.01,port=base+t2,pool=t2+rush+base",
        "hard" => "cadence=0.85,h=16,screen=1,samples=3,k=4,cands=60,waits=0/3/7,feeders=3,refine=1,pess=0.4,self=t2",
        "grandmaster" => "cadence=0.5,h=18,screen=2,samples=4,k=5,cands=90,waits=0/3/7,feeders=4,refine=1",
        "nightmare" => "cadence=0.3",
        "nightmare+" => "cadence=0.2,samples=8,k=8",
        _ => return None,
    })
}

/// Build a planner from `plan`, `plan:<tier>`, `plan:key=value,...` or `plan:<tier>,key=value,...`,
/// with an optional `@cadence` suffix.
pub fn make_planner(spec: &str) -> Result<crate::planner::Planner, String> {
    use crate::planner::{PlanParams, Planner};
    let (body, cadence) = match spec.split_once('@') {
        Some((b, c)) => (b, Some(c.parse::<f64>().map_err(|_| format!("bad cadence {c}"))?)),
        None => (spec, None),
    };
    let mut p = PlanParams::default();
    if let Some((_, opts)) = body.split_once(':') {
        for item in opts.split(',').filter(|x| !x.is_empty()) {
            match item.split_once('=') {
                Some((k, v)) => p.set(k, v)?,
                None => {
                    let tier = plan_tier(item).ok_or(format!("unknown plan tier {item}"))?;
                    for kv in tier.split(',') {
                        let (k, v) = kv.split_once('=').unwrap();
                        p.set(k, v)?;
                    }
                }
            }
        }
    }
    if let Some(c) = cadence {
        p.cadence = c;
    }
    Ok(Planner::new(spec, p))
}

pub const SCRIPTED: [&str; 5] = ["base", "rush", "econ", "farmer", "turtle"];
