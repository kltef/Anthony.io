//! Play one game of an agent against a scripted opponent and print what happens.
//!   usage: watch [agent=plan:nightmare] [opponent=rush] [seed=1] [seat=1] [verbose_until=20] [verbose_from=0]

use stateio::agent::TICK;
use stateio::rules::*;
use stateio::setup::{new_game, Setup};
use stateio::{roster, Map, Rng};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let me_spec = a.get(0).cloned().unwrap_or("plan:nightmare".into());
    let opp_spec = a.get(1).cloned().unwrap_or("rush".into());
    let seed: u64 = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(1);
    let seat: u8 = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(1);
    let verbose_until: f64 = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(20.0);
    let verbose_from: f64 = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(0.0);
    let map = Map::continental();
    let w = World::new(map.clone(), Rules::standard(&map, 1.0));
    let mut s = new_game(&w.map, &Setup::duel(), &mut Rng::new(seed));
    s.recompute_supply(&w);
    let mut me = roster::make(&me_spec);
    let mut opp = roster::make(&opp_spec);
    let other = 3 - seat;
    me.reset(&w, seat, seed);
    opp.reset(&w, other, seed + 1);
    let (mut ba, mut bb) = (Vec::new(), Vec::new());
    let mut next_print = 0.0;
    let name = |i: u8| w.map.names[i as usize].clone();
    let t0 = std::time::Instant::now();
    while s.winner().is_none() && s.time < 600.0 {
        ba.clear();
        bb.clear();
        let before = me.stats().0;
        me.act(&w, &s, &mut ba);
        opp.act(&w, &s, &mut bb);
        if me.stats().0 > before && s.time < verbose_until && s.time >= verbose_from && !ba.is_empty() {
            println!("  t={:6.2} {}", s.time, me.log().unwrap_or_default());
            for act in &ba {
                match *act {
                    Action::Send { src, dst, amount } => println!(
                        "      send {:>4} {} ({:.0}) -> {} (owner {} troops {:.0})",
                        if amount >= 1e8 { "all".to_string() } else { format!("{amount:.0}") }, name(src), s.troops[src as usize], name(dst), s.owner[dst as usize], s.troops[dst as usize]
                    ),
                    Action::Invest(i) => println!("      invest {}", name(i)),
                    Action::Upgrade(i) => println!("      upgrade {}", name(i)),
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
            println!("t={:5.0}  {}: {:2} states {:5.0} troops | {}: {:2} states {:5.0} troops | orbs {}", s.time, me_spec, c1, t1, opp_spec, c2, t2, s.orbs.len());
        }
    }
    let (d, r) = me.stats();
    println!("winner: {:?} at t={:.0}  ({} is owner {})  decisions {}  rollouts {}  wall {:.1}s", s.winner(), s.time, me_spec, seat, d, r, t0.elapsed().as_secs_f64());
}
