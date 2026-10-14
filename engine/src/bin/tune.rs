//! Evolve scripted policies by league self-play (cross-entropy method).
//!
//! Each round finds the policy that scores best against the current league, then adds it to the
//! league, so later champions have to beat earlier ones as well as the hand-written styles.
//!   usage: tune [--rounds N] [--gens N] [--pop N] [--boards N] [--prefix NAME] [--seed S] [--league a,b,c]
//! Prints `name: key=value ...` lines for data/policies.txt.

use stateio::agent::{play, Agent, Scripted};
use stateio::policy::{PolicyParams, FIELDS};
use stateio::rules::*;
use stateio::setup::{new_game, Setup};
use stateio::{roster, Map, Rng};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

fn score(world: &World, starts: &[State], a: &PolicyParams, b: &PolicyParams) -> f64 {
    let mut pts = 0.0;
    for (k, st) in starts.iter().enumerate() {
        for seat in 0..2 {
            let pa: Box<dyn Agent> = Box::new(Scripted::new("a", a.clone(), 0.5));
            let pb: Box<dyn Agent> = Box::new(Scripted::new("b", b.clone(), 0.5));
            let mut agents = if seat == 0 { vec![pa, pb] } else { vec![pb, pa] };
            let o = play(world, st, &mut agents, 480.0, k as u64 * 31 + seat as u64);
            let a_owner = if seat == 0 { 1 } else { 2 };
            pts += match o.winner {
                Some(w) if w == a_owner => 1.0,
                Some(_) => 0.0,
                None => 0.5,
            };
        }
    }
    pts / (2 * starts.len()) as f64
}

fn gauss(r: &mut Rng) -> f64 {
    let (u, v) = (r.f64().max(1e-12), r.f64());
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut rounds, mut gens, mut pop, mut boards, mut prefix, mut seed) = (4usize, 14usize, 48usize, 30usize, "t".to_string(), 7u64);
    let mut league_names: Vec<String> = vec!["rush".into(), "base".into(), "econ".into()];
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--rounds" => rounds = args[i + 1].parse().unwrap(),
            "--gens" => gens = args[i + 1].parse().unwrap(),
            "--pop" => pop = args[i + 1].parse().unwrap(),
            "--boards" => boards = args[i + 1].parse().unwrap(),
            "--prefix" => prefix = args[i + 1].clone(),
            "--seed" => seed = args[i + 1].parse().unwrap(),
            "--league" => league_names = args[i + 1].split(',').map(|s| s.to_string()).collect(),
            a => panic!("unknown argument {a}"),
        }
        i += 2;
    }
    let map = Map::continental();
    let world = World::new(map.clone(), Rules::standard(&map, 1.0));
    let setup = Setup::duel();
    let mut league: Vec<(String, PolicyParams)> = league_names.iter().map(|n| (n.clone(), roster::policy(n).unwrap_or_else(|| panic!("unknown policy {n}")))).collect();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let d = FIELDS.len();
    let mut rng = Rng::new(seed);

    for round in 0..rounds {
        // start each round from the strongest league member so far
        let mut mean = league.last().unwrap().1.to_vec();
        let mut sigma: Vec<f64> = FIELDS.iter().map(|f| (f.2 - f.1) * 0.25).collect();
        let mut best: (f64, Vec<f64>) = (-1.0, mean.clone());
        for gen in 0..gens {
            let starts: Vec<State> = (0..boards).map(|b| new_game(&world.map, &setup, &mut Rng::new(seed * 7919 + (round * 1000 + gen * 50 + b) as u64))).collect();
            let mut cands: Vec<Vec<f64>> = (0..pop)
                .map(|_| (0..d).map(|k| (mean[k] + sigma[k] * gauss(&mut rng)).clamp(FIELDS[k].1, FIELDS[k].2)).collect())
                .collect();
            cands[0] = mean.clone();
            let fit = Mutex::new(vec![0.0f64; pop]);
            let next = AtomicUsize::new(0);
            std::thread::scope(|sc| {
                for _ in 0..threads {
                    sc.spawn(|| loop {
                        let c = next.fetch_add(1, Ordering::Relaxed);
                        if c >= pop {
                            break;
                        }
                        let p = PolicyParams::default().with_vec(&cands[c]);
                        let (mut sum, mut worst) = (0.0, 1.0f64);
                        for (_, opp) in &league {
                            let s = score(&world, &starts, &p, opp);
                            sum += s;
                            worst = worst.min(s);
                        }
                        // reward beating everyone, not just farming the weakest opponent
                        fit.lock().unwrap()[c] = 0.5 * sum / league.len() as f64 + 0.5 * worst;
                    });
                }
            });
            let fit = fit.into_inner().unwrap();
            let mut order: Vec<usize> = (0..pop).collect();
            order.sort_by(|a, b| fit[*b].partial_cmp(&fit[*a]).unwrap());
            let elite = &order[..pop / 6];
            for k in 0..d {
                let m = elite.iter().map(|&c| cands[c][k]).sum::<f64>() / elite.len() as f64;
                let var = elite.iter().map(|&c| (cands[c][k] - m).powi(2)).sum::<f64>() / elite.len() as f64;
                mean[k] = m;
                sigma[k] = (var.sqrt() * 1.1).max((FIELDS[k].2 - FIELDS[k].1) * 0.01);
            }
            if fit[order[0]] >= best.0 {
                best = (fit[order[0]], cands[order[0]].clone());
            }
            eprintln!("round {round} gen {gen:2}: best {:.3}  mean-of-elite {:.3}", fit[order[0]], elite.iter().map(|&c| fit[c]).sum::<f64>() / elite.len() as f64);
        }
        // the distribution mean is less noisy than any single lucky sample
        let champ = PolicyParams::default().with_vec(&mean);
        let name = format!("{prefix}{round}");
        let starts: Vec<State> = (0..200).map(|b| new_game(&world.map, &setup, &mut Rng::new(99_000 + b as u64))).collect();
        let vs: Vec<String> = league.iter().map(|(n, o)| format!("{} {:.0}%", n, 100.0 * score(&world, &starts, &champ, o))).collect();
        eprintln!("== {name}: {}", vs.join(", "));
        println!("{name}: {}", champ.to_text());
        league.push((name, champ));
    }
}
