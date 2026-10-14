//! Head-to-head arena. Every board is played twice with the seats swapped, so map luck cancels.
//!   usage: arena <agentA> <agentB> [--games N] [--map NAME] [--threads N] [--seed S] [--max-time T]

use stateio::agent::{play, Agent};
use stateio::rules::*;
use stateio::setup::{new_game, Setup};
use stateio::{roster, Map, Rng};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

#[derive(Default, Clone)]
struct Tally {
    win: u32,
    draw: u32,
    loss: u32,
    time: f64,
    timeouts: u32,
    orders: [u64; 2],
    /// boards where A won from both seats / B won from both seats
    sweep: [u32; 2],
    rejected: [u64; 2],
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pos = Vec::new();
    let (mut games, mut map_name, mut threads, mut seed, mut max_time) = (100usize, "Continental U.S.".to_string(), 0usize, 1u64, 600.0f64);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => { games = args[i + 1].parse().unwrap(); i += 1; }
            "--map" => { map_name = args[i + 1].clone(); i += 1; }
            "--threads" => { threads = args[i + 1].parse().unwrap(); i += 1; }
            "--seed" => { seed = args[i + 1].parse().unwrap(); i += 1; }
            "--max-time" => { max_time = args[i + 1].parse().unwrap(); i += 1; }
            a => pos.push(a.to_string()),
        }
        i += 1;
    }
    if pos.len() < 2 {
        eprintln!("usage: arena <agentA> <agentB> [--games N] [--map NAME] [--threads N] [--seed S] [--max-time T]");
        std::process::exit(2);
    }
    if threads == 0 {
        threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    }
    let map = Map::builtin().into_iter().find(|m| m.name == map_name).expect("unknown map");
    let world = World::new(map.clone(), Rules::standard(&map, 1.0));
    let setup = Setup::duel();
    let boards = (games + 1) / 2;
    let next = AtomicUsize::new(0);
    let total = Mutex::new(Tally::default());
    let t0 = std::time::Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                let mut local = Tally::default();
                loop {
                    let b = next.fetch_add(1, Ordering::Relaxed);
                    if b >= boards {
                        break;
                    }
                    let bseed = seed.wrapping_mul(1_000_003).wrapping_add(b as u64);
                    let start = new_game(&world.map, &setup, &mut Rng::new(bseed));
                    let mut a_wins = 0;
                    let mut b_wins = 0;
                    for seat in 0..2 {
                        // seat 0: A is owner 1. seat 1: A is owner 2.
                        let mut agents: Vec<Box<dyn Agent>> = if seat == 0 { vec![roster::make(&pos[0]), roster::make(&pos[1])] } else { vec![roster::make(&pos[1]), roster::make(&pos[0])] };
                        let o = play(&world, &start, &mut agents, max_time, bseed ^ (seat as u64) << 40);
                        let a_owner = if seat == 0 { 1u8 } else { 2u8 };
                        match o.winner {
                            Some(w) if w == a_owner => {
                                local.win += 1;
                                a_wins += 1;
                            }
                            Some(_) => {
                                local.loss += 1;
                                b_wins += 1;
                            }
                            None => local.draw += 1,
                        }
                        local.time += o.time;
                        local.timeouts += o.timed_out as u32;
                        let b_owner = 3 - a_owner;
                        local.orders[0] += o.orders[a_owner as usize] as u64;
                        local.orders[1] += o.orders[b_owner as usize] as u64;
                        local.rejected[0] += o.rejected[a_owner as usize] as u64;
                        local.rejected[1] += o.rejected[b_owner as usize] as u64;
                    }
                    local.sweep[0] += (a_wins == 2) as u32;
                    local.sweep[1] += (b_wins == 2) as u32;
                }
                let mut t = total.lock().unwrap();
                t.win += local.win;
                t.draw += local.draw;
                t.loss += local.loss;
                t.time += local.time;
                t.timeouts += local.timeouts;
                for k in 0..2 {
                    t.sweep[k] += local.sweep[k];
                    t.orders[k] += local.orders[k];
                    t.rejected[k] += local.rejected[k];
                }
            });
        }
    });
    let t = total.lock().unwrap().clone();
    let n = (t.win + t.draw + t.loss) as f64;
    let score = (t.win as f64 + 0.5 * t.draw as f64) / n;
    let se = (score * (1.0 - score) / n).sqrt();
    println!(
        "{} vs {}: {}W {}D {}L  score {:.1}% (+/- {:.1})  boards swept {}-{} of {}  avg {:.0}s  timeouts {}  orders/game {:.0} vs {:.0}  rejected {:.1}% vs {:.1}%  [{:.1}s wall]",
        pos[0], pos[1], t.win, t.draw, t.loss, 100.0 * score, 196.0 * se, t.sweep[0], t.sweep[1], boards, t.time / n, t.timeouts,
        t.orders[0] as f64 / n, t.orders[1] as f64 / n,
        100.0 * t.rejected[0] as f64 / t.orders[0].max(1) as f64, 100.0 * t.rejected[1] as f64 / t.orders[1].max(1) as f64,
        t0.elapsed().as_secs_f64()
    );
}
