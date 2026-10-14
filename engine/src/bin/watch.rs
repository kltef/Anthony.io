//! Play one game of the search agent against a scripted opponent and print what happens.
//!   usage: watch [opponent=rush] [seed=1] [search_seat=1] [verbose_until=20]

use stateio::agent::{Agent, TICK};
use stateio::rules::*;
use stateio::search::{SearchParams, Searcher};
use stateio::setup::{new_game, Setup};
use stateio::{roster, Map, Rng};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let opp_spec = a.get(0).cloned().unwrap_or("rush".into());
    let seed: u64 = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(1);
    let seat: u8 = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(1);
    let verbose_until: f64 = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(20.0);
    let verbose_from: f64 = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(0.0);
    let map = Map::continental();
    let w = World::new(map.clone(), Rules::standard(&map, 1.0));
    let mut s = new_game(&w.map, &Setup::duel(), &mut Rng::new(seed));
    s.recompute_supply(&w);
    let mut me = Searcher::new("search", SearchParams::default());
    let mut opp = roster::make(&opp_spec);
    let other = 3 - seat;
    me.reset(&w, seat, seed);
    opp.reset(&w, other, seed + 1);
    let (mut ba, mut bb) = (Vec::new(), Vec::new());
    let mut next_print = 0.0;
    let t0 = std::time::Instant::now();
    while s.winner().is_none() && s.time < 600.0 {
        ba.clear();
        bb.clear();
        let before = me.decisions;
        me.act(&w, &s, &mut ba);
        opp.act(&w, &s, &mut bb);
        if me.decisions > before && s.time < verbose_until && s.time >= verbose_from && !me.last.chosen.is_empty() {
            println!("  t={:6.2} wait={:+.4} cands={} rollouts={}", s.time, me.last.wait_value, me.last.candidates, me.last.rollouts);
            for (act, v) in &me.last.chosen {
                match *act {
                    Action::Send { src, dst, amount } => println!(
                        "      send {:>3} {} ({:.0}) -> {} (owner {} troops {:.0})  value {:+.4}",
                        amount, w.map.names[src as usize], s.troops[src as usize], w.map.names[dst as usize], s.owner[dst as usize], s.troops[dst as usize], v
                    ),
                    Action::Invest(i) => println!("      invest {}  value {:+.4}", w.map.names[i as usize], v),
                    Action::Upgrade(i) => println!("      upgrade {}  value {:+.4}", w.map.names[i as usize], v),
                }
            }
        }
        for &x in &ba {
            s.apply(&w, seat, x);
        }
        for &x in &bb {
            s.apply(&w, other, x);
        }
        s.step(&w, TICK);
        if s.time >= next_print {
            next_print += 10.0;
            let (t1, c1) = s.totals(seat);
            let (t2, c2) = s.totals(other);
            println!("t={:5.0}  search: {:2} states {:5.0} troops | {}: {:2} states {:5.0} troops | orbs {}", s.time, c1, t1, opp_spec, c2, t2, s.orbs.len());
        }
    }
    println!(
        "winner: {:?} at t={:.0}  (search is owner {})  decisions {}  rollouts {}  {:.1} us/rollout  wall {:.1}s",
        s.winner(), s.time, seat, me.decisions, me.total_rollouts,
        t0.elapsed().as_secs_f64() * 1e6 / me.total_rollouts.max(1) as f64, t0.elapsed().as_secs_f64()
    );
}
