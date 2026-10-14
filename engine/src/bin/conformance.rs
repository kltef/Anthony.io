//! Replays traces written by `tools/conformance_gen.js` (the original JavaScript simulation)
//! through the Rust engine and reports any difference.
//!   usage: cargo run --release --bin conformance [dir=target/conformance]

use stateio::rules::*;
use stateio::Map;
use std::fs;

fn f(s: &str) -> f64 {
    s.parse().unwrap_or_else(|_| panic!("bad number {s}"))
}

fn run(path: &std::path::Path, maps: &[Map]) -> Result<(usize, usize), String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut lines = text.lines().enumerate().peekable();
    let mut world: Option<World> = None;
    let mut st = State::empty(0);
    let (mut checks, mut ops) = (0, 0);
    while let Some((ln, line)) = lines.next() {
        let p: Vec<&str> = line.split('|').collect();
        match p[0] {
            "W" => {
                let map = maps.iter().find(|m| m.name == p[1]).ok_or("unknown map")?.clone();
                let mut rules = Rules::standard(&map, 1.0);
                rules.orb_speed = f(p[2]);
                rules.depth = p[3] == "1";
                rules.adjacency = p[4] == "1";
                rules.terrain = p[5] == "1";
                st = State::empty(map.n);
                world = Some(World::new(map, rules));
            }
            "G" => {
                let w = world.as_mut().unwrap();
                w.rules.grow = [0.0; MAX_OWNERS];
                for (o, v) in p[1..].iter().enumerate() {
                    w.rules.grow[o] = f(v);
                }
            }
            "I" => {
                let i: usize = p[1].parse().unwrap();
                st.owner[i] = p[2].parse().unwrap();
                st.troops[i] = f(p[3]);
                st.tile[i] = p[4].parse().unwrap();
                st.off[i] = p[5] == "1";
            }
            "S" => {
                st.step(world.as_ref().unwrap(), f(p[1]));
                ops += 1;
            }
            "A" => {
                let ok = st.send(world.as_ref().unwrap(), p[1].parse().unwrap(), p[2].parse().unwrap(), p[3].parse().unwrap(), f(p[4]));
                if ok != (p[5] == "1") {
                    return Err(format!("line {}: send accepted={} but JS accepted={}: {}", ln + 1, ok, p[5], line));
                }
                ops += 1;
            }
            "V" | "U" => {
                let w = world.as_ref().unwrap();
                let (o, i) = (p[1].parse().unwrap(), p[2].parse().unwrap());
                let ok = if p[0] == "V" { st.invest(w, o, i) } else { st.upgrade(w, o, i) };
                if ok != (p[3] == "1") {
                    return Err(format!("line {}: build accepted={} but JS accepted={}: {}", ln + 1, ok, p[3], line));
                }
                ops += 1;
            }
            "C" => {
                if (st.time - f(p[1])).abs() > 1e-6 {
                    return Err(format!("line {}: time {} vs JS {}", ln + 1, st.time, p[1]));
                }
                let orbs: usize = p[2].parse().unwrap();
                if st.orbs.len() != orbs {
                    return Err(format!("line {}: {} orbs in flight vs JS {} at t={:.3}", ln + 1, st.orbs.len(), orbs, st.time));
                }
                for i in 0..st.n {
                    let (sl, sline) = lines.next().ok_or("truncated checkpoint")?;
                    let q: Vec<&str> = sline.split('|').collect();
                    let bad = st.owner[i] != q[1].parse::<u8>().unwrap()
                        || (st.troops[i] - f(q[2])).abs() > 1e-6
                        || (st.grow_acc[i] - f(q[3])).abs() > 1e-6
                        || st.level[i] != q[4].parse::<u8>().unwrap()
                        || st.inv[i] != q[5].parse::<u8>().unwrap()
                        || st.off[i] != (q[6] == "1");
                    if bad {
                        return Err(format!(
                            "line {}: state {} at t={:.3}: rust owner={} troops={} acc={} lvl={} inv={} off={} | JS {}",
                            sl + 1, i, st.time, st.owner[i], st.troops[i], st.grow_acc[i], st.level[i], st.inv[i], st.off[i], sline
                        ));
                    }
                }
                checks += 1;
            }
            _ => return Err(format!("line {}: unknown record {}", ln + 1, line)),
        }
    }
    Ok((checks, ops))
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "target/conformance".to_string());
    let maps = Map::builtin();
    let mut files: Vec<_> = fs::read_dir(&dir).expect("trace dir").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().map_or(false, |x| x == "txt")).collect();
    files.sort();
    let (mut pass, mut fail, mut checks, mut ops) = (0, 0, 0, 0);
    for p in &files {
        match run(p, &maps) {
            Ok((c, o)) => {
                pass += 1;
                checks += c;
                ops += o;
            }
            Err(e) => {
                fail += 1;
                println!("FAIL {}: {}", p.file_name().unwrap().to_string_lossy(), e);
            }
        }
    }
    println!("conformance: {pass} traces passed, {fail} failed ({checks} checkpoints, {ops} operations)");
    if fail > 0 {
        std::process::exit(1);
    }
}
