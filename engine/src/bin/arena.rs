//! Head-to-head arena. Every board is played twice with the seats swapped, so map luck cancels.
//!   usage: arena <agentA> <agentB> [--games N] [--map NAME] [--threads N] [--seed S] [--max-time T]

use stateio::agent::{play, Agent};
use stateio::rules::*;
use stateio::setup::{new_game, Setup};
use stateio::{roster, Map, Rng};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Wraps agent A and measures how it moves troops: the share sent onto its own states in big
/// moves (30+), stacks sent A->B then straight back B->A within 12 s, and states held at 60 s.
struct Probe {
    inner: Box<dyn Agent>,
    me: u8,
    big: Vec<(f64, u8, u8)>,
    own: f64,
    total: f64,
    st60: Option<u32>,
}

impl Agent for Probe {
    fn name(&self) -> String {
        self.inner.name()
    }
    fn reset(&mut self, w: &World, me: u8, seed: u64) {
        self.me = me;
        self.inner.reset(w, me, seed);
    }
    fn act(&mut self, w: &World, s: &State, out: &mut Vec<Action>) {
        let n0 = out.len();
        self.inner.act(w, s, out);
        let mut left = s.troops.clone();
        for a in &out[n0..] {
            if let Action::Send { src, dst, amount } = *a {
                let amt = amount.floor().min(left[src as usize].floor()).max(0.0);
                left[src as usize] -= amt;
                self.total += amt;
                if s.owner[dst as usize] == self.me && amt >= 30.0 {
                    self.own += amt;
                    self.big.push((s.time, src, dst));
                }
            }
        }
        if self.st60.is_none() && s.time >= 60.0 {
            self.st60 = Some(s.owned_mask(self.me).count_ones());
        }
    }
}

impl Probe {
    /// 30+ stacks sent onto a state of mine that itself sent a 30+ stack on within 12 s (loops of
    /// any length: A->B->A, A->B->C->A, ...).
    fn returns(&self) -> u32 {
        let mut n = 0;
        for (i, b) in self.big.iter().enumerate() {
            if self.big[..i].iter().any(|a| b.0 - a.0 <= 12.0 && a.1 == b.2) {
                n += 1;
            }
        }
        n
    }
    fn bounces(&self) -> u32 {
        let mut n = 0;
        for (i, a) in self.big.iter().enumerate() {
            for b in &self.big[i + 1..] {
                if b.0 - a.0 > 12.0 {
                    break;
                }
                if b.1 == a.2 && b.2 == a.1 {
                    n += 1;
                    break;
                }
            }
        }
        n
    }
}

#[derive(Default, Clone)]
struct Tally {
    probe_games: u32,
    probe_own: f64,
    probe_total: f64,
    probe_bounce: u64,
    probe_return: u64,
    probe_st60: u64,
    probe_st60_n: u32,
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
                        let a_agent: Box<dyn Agent> = Box::new(Probe { inner: roster::make(&pos[0]), me: 0, big: Vec::new(), own: 0.0, total: 0.0, st60: None });
                        let mut agents: Vec<Box<dyn Agent>> = if seat == 0 { vec![a_agent, roster::make(&pos[1])] } else { vec![roster::make(&pos[1]), a_agent] };
                        let o = play(&world, &start, &mut agents, max_time, bseed ^ (seat as u64) << 40);
                        {
                            // read the probe back (it is the agent in A's seat)
                            let pa: &Box<dyn Agent> = &agents[seat];
                            let pr = unsafe { &*(pa.as_ref() as *const dyn Agent as *const Probe) };
                            local.probe_games += 1;
                            local.probe_own += pr.own;
                            local.probe_total += pr.total;
                            local.probe_bounce += pr.bounces() as u64;
                            local.probe_return += pr.returns() as u64;
                            if let Some(c) = pr.st60 {
                                local.probe_st60 += c as u64;
                                local.probe_st60_n += 1;
                            }
                        }
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
                t.probe_games += local.probe_games;
                t.probe_own += local.probe_own;
                t.probe_total += local.probe_total;
                t.probe_bounce += local.probe_bounce;
                t.probe_return += local.probe_return;
                t.probe_st60 += local.probe_st60;
                t.probe_st60_n += local.probe_st60_n;
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
    println!(
        "   A moves: {:.0}% of troops sent onto its own states in 30+ moves, {:.1} back-and-forth bounces/game, {:.1} loop returns/game, {:.1} states at 60 s",
        100.0 * t.probe_own / t.probe_total.max(1.0), t.probe_bounce as f64 / t.probe_games.max(1) as f64, t.probe_return as f64 / t.probe_games.max(1) as f64, t.probe_st60 as f64 / t.probe_st60_n.max(1) as f64
    );
}
