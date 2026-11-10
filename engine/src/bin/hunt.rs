//! The exploit hunter: evolve opponent styles to beat a target AI (cross-entropy method), so its
//! holes turn up here before a player finds them.
//!
//! A style is a vector of dials. `--family chris` evolves the play-tester bot (feed / funnel /
//! margins / timing / half sends); `--family policy` evolves the scripted policy's 12 weights.
//! Each generation samples `--pop` styles around the current mean, plays each against the target
//! on `--boards` boards from both seats, keeps the best sixth and moves the mean toward them.
//!
//! Fitness = win rate + 0.5 x the style's average share of material at 60 / 90 / 120 s (0 once
//! dead), so the search has a slope to climb even while the target wins every game.
//!
//!   usage: hunt [--target plan:nightmare+] [--family chris|policy] [--gens N] [--pop N]
//!               [--boards N] [--seed S] [--out data/exploits.txt]
//! Every generation prints its best; the final champion (and its win rate on 100 fresh boards) is
//! appended to `--out` as `name | family | win% | spec`, ready for the regression suite.

use stateio::agent::{strength, Agent, Scripted, TICK};
use stateio::chris::{Chris, ChrisParams};
use stateio::policy::{PolicyParams, FIELDS};
use stateio::rules::*;
use stateio::setup::{new_game, Setup};
use stateio::{roster, Map, Rng};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// (name, lo, hi) of each chris dial. Booleans are dials too: on when > 0.5.
const CHRIS: [(&str, f64, f64); 9] = [
    ("cadence", 0.3, 2.0),
    ("nm", 1.0, 3.0),
    ("er", 1.0, 10.0),
    ("feed", 1.0, 10.0),
    ("slack", 0.0, 60.0),
    ("joint", 0.0, 1.0),
    ("half", 0.0, 1.0),
    ("after", 0.0, 120.0),
    ("funnel", 0.0, 1.0),
];

#[derive(Clone, Copy, PartialEq)]
enum Family {
    Chris,
    Policy,
}

fn dims(f: Family) -> Vec<(&'static str, f64, f64)> {
    match f {
        Family::Chris => CHRIS.to_vec(),
        Family::Policy => FIELDS.to_vec(),
    }
}

fn chris_params(v: &[f64]) -> ChrisParams {
    ChrisParams {
        cadence: v[0],
        neutral_margin: v[1],
        enemy_ratio: v[2],
        min_feed: v[3],
        hub_slack: v[4],
        joint: v[5] > 0.5,
        send_half: v[6] > 0.5,
        enemy_after: v[7],
        funnel: v[8] > 0.5,
    }
}

/// The roster spec for a style (chris), or a key=value description (policy).
fn spec(f: Family, v: &[f64]) -> String {
    match f {
        Family::Chris => {
            let p = chris_params(v);
            format!(
                "chris:nm={:.2},er={:.2},feed={:.1},slack={:.0},joint={},half={},after={:.0},funnel={}@{:.2}",
                p.neutral_margin, p.enemy_ratio, p.min_feed, p.hub_slack, p.joint as u8, p.send_half as u8, p.enemy_after, p.funnel as u8, p.cadence
            )
        }
        Family::Policy => format!("policy {}", PolicyParams::default().with_vec(v).to_text()),
    }
}

fn agent(f: Family, v: &[f64]) -> Box<dyn Agent> {
    match f {
        Family::Chris => {
            let p = chris_params(v);
            Box::new(Chris::new("hunter", p))
        }
        Family::Policy => Box::new(Scripted::new("hunter", PolicyParams::default().with_vec(v), 0.5)),
    }
}

/// One game; returns (won, material share at 60/90/120 s averaged).
fn game(w: &World, start: &State, target: &str, f: Family, v: &[f64], seat: usize, seed: u64) -> (f64, f64) {
    let mut s = start.clone();
    let mut agents: Vec<Box<dyn Agent>> = if seat == 0 { vec![agent(f, v), roster::make(target)] } else { vec![roster::make(target), agent(f, v)] };
    let me = (seat + 1) as u8;
    for (k, a) in agents.iter_mut().enumerate() {
        a.reset(w, (k + 1) as u8, seed.wrapping_add(k as u64 * 977));
    }
    s.recompute_supply(w);
    let mut buf: Vec<Vec<Action>> = vec![Vec::new(), Vec::new()];
    let checks = [60.0, 90.0, 120.0];
    let (mut ci, mut share) = (0usize, 0.0);
    let mut tick = 0u64;
    loop {
        for k in 0..2 {
            buf[k].clear();
            if s.alive() >> (k + 1) & 1 == 1 {
                agents[k].act(w, &s, &mut buf[k]);
            }
        }
        for r in 0..2 {
            let k = (r + tick as usize) % 2;
            for &a in &buf[k] {
                s.apply(w, (k + 1) as u8, a);
            }
        }
        s.step(w, TICK);
        tick += 1;
        while ci < 3 && s.time >= checks[ci] {
            let a = if s.alive() >> me & 1 == 1 { strength(w, &s, me) } else { 0.0 };
            let b = strength(w, &s, 3 - me);
            share += a / (a + b).max(1.0);
            ci += 1;
        }
        if let Some(win) = s.winner() {
            // a finished game counts its final share for the checks it did not reach
            let fin = if win == me { 1.0 } else { 0.0 };
            share += fin * (3 - ci) as f64;
            return (fin, share / 3.0);
        }
        if s.time >= 300.0 {
            let a = strength(w, &s, me);
            let b = strength(w, &s, 3 - me);
            return (if a > 1.5 * b { 1.0 } else { 0.0 }, share / 3.0);
        }
    }
}

fn gauss(r: &mut Rng) -> f64 {
    let (u, v) = (r.f64().max(1e-12), r.f64());
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut target, mut family, mut gens, mut pop, mut boards, mut seed, mut out) =
        ("plan:nightmare+".to_string(), Family::Chris, 12usize, 36usize, 8usize, 11u64, "data/exploits.txt".to_string());
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--target" => target = args[i + 1].clone(),
            "--family" => family = if args[i + 1] == "policy" { Family::Policy } else { Family::Chris },
            "--gens" => gens = args[i + 1].parse().unwrap(),
            "--pop" => pop = args[i + 1].parse().unwrap(),
            "--boards" => boards = args[i + 1].parse().unwrap(),
            "--seed" => seed = args[i + 1].parse().unwrap(),
            "--out" => out = args[i + 1].clone(),
            a => panic!("unknown argument {a}"),
        }
        i += 2;
    }
    let map = Map::continental();
    let world = World::new(map.clone(), Rules::standard(&map, 1.0));
    let setup = Setup::duel();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let dm = dims(family);
    let d = dm.len();
    let mut rng = Rng::new(seed);
    // start from the known styles: the fitted Chris bot, or the tuned policy
    let mut mean: Vec<f64> = match family {
        Family::Chris => vec![0.5, 1.3, 6.0, 2.0, 20.0, 1.0, 1.0, 0.0, 0.5],
        Family::Policy => roster::policy("t4").unwrap_or_default().to_vec(),
    };
    let mut sigma: Vec<f64> = dm.iter().map(|f| (f.2 - f.1) * 0.3).collect();
    let mut best: (f64, Vec<f64>) = (-1.0, mean.clone());
    let t0 = std::time::Instant::now();
    for gen in 0..gens {
        let starts: Vec<State> = (0..boards).map(|b| new_game(&world.map, &setup, &mut Rng::new(seed * 7919 + (gen * 100 + b) as u64))).collect();
        let mut cands: Vec<Vec<f64>> = (0..pop).map(|_| (0..d).map(|k| (mean[k] + sigma[k] * gauss(&mut rng)).clamp(dm[k].1, dm[k].2)).collect()).collect();
        cands[0] = mean.clone();
        // jobs = (candidate, board, seat): spread over every core
        let jobs = pop * boards * 2;
        let res = Mutex::new(vec![(0.0f64, 0.0f64); pop]);
        let next = AtomicUsize::new(0);
        std::thread::scope(|sc| {
            for _ in 0..threads {
                sc.spawn(|| loop {
                    let j = next.fetch_add(1, Ordering::Relaxed);
                    if j >= jobs {
                        break;
                    }
                    let (c, b, seat) = (j / (boards * 2), (j / 2) % boards, j % 2);
                    let (won, share) = game(&world, &starts[b], &target, family, &cands[c], seat, (gen * 1000 + b) as u64 ^ (seat as u64) << 40);
                    let mut r = res.lock().unwrap();
                    r[c].0 += won;
                    r[c].1 += share;
                });
            }
        });
        let n = (boards * 2) as f64;
        let fit: Vec<f64> = res.into_inner().unwrap().iter().map(|&(w, s)| w / n + 0.5 * s / n).collect();
        let mut order: Vec<usize> = (0..pop).collect();
        order.sort_by(|a, b| fit[*b].partial_cmp(&fit[*a]).unwrap());
        let elite = &order[..(pop / 6).max(2)];
        for k in 0..d {
            let m = elite.iter().map(|&c| cands[c][k]).sum::<f64>() / elite.len() as f64;
            let var = elite.iter().map(|&c| (cands[c][k] - m).powi(2)).sum::<f64>() / elite.len() as f64;
            mean[k] = m;
            sigma[k] = (var.sqrt() * 1.1).max((dm[k].2 - dm[k].1) * 0.02);
        }
        if fit[order[0]] >= best.0 {
            best = (fit[order[0]], cands[order[0]].clone());
        }
        eprintln!(
            "gen {gen:2}: best fitness {:.3}  elite mean {:.3}  [{:.0}s]  best: {}",
            fit[order[0]],
            elite.iter().map(|&c| fit[c]).sum::<f64>() / elite.len() as f64,
            t0.elapsed().as_secs_f64(),
            spec(family, &cands[order[0]])
        );
    }
    // judge the mean and the single best on fresh boards; keep the stronger
    let fresh: Vec<State> = (0..50).map(|b| new_game(&world.map, &setup, &mut Rng::new(77_000 + b as u64))).collect();
    let judge = |v: &Vec<f64>| -> (f64, f64) {
        let tot = Mutex::new((0.0, 0.0));
        let next = AtomicUsize::new(0);
        std::thread::scope(|sc| {
            for _ in 0..threads {
                sc.spawn(|| loop {
                    let j = next.fetch_add(1, Ordering::Relaxed);
                    if j >= fresh.len() * 2 {
                        break;
                    }
                    let (w, s) = game(&world, &fresh[j / 2], &target, family, v, j % 2, 5_000 + j as u64);
                    let mut t = tot.lock().unwrap();
                    t.0 += w;
                    t.1 += s;
                });
            }
        });
        let t = tot.into_inner().unwrap();
        let n = (fresh.len() * 2) as f64;
        (t.0 / n, t.1 / n)
    };
    let (wm, sm) = judge(&mean);
    let (wb, sb) = judge(&best.1);
    let (champ, win, share) = if wm + 0.5 * sm >= wb + 0.5 * sb { (mean.clone(), wm, sm) } else { (best.1.clone(), wb, sb) };
    let line = format!(
        "hunt{} | {} | {:.1}% | share {:.2} | {} | vs {}",
        seed,
        if family == Family::Chris { "chris" } else { "policy" },
        100.0 * win,
        share,
        spec(family, &champ),
        target
    );
    println!("{line}");
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&out) {
        let _ = writeln!(f, "{line}");
    }
}
