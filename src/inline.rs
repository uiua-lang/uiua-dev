//! Logic for inlining code before running

use ecow::EcoVec;
use enum_iterator::{all, Sequence};

use crate::{
    algorithm::invert::under_instrs, Assembly, FuncSlice, Function, ImplPrimitive, Instr, Primitive,
};

const DEBUG: bool = false;

macro_rules! dbgw {
    ($($arg:tt)*) => {
        if DEBUG {
            print!($($arg)*); // Allow println
        }
    }
}

macro_rules! dbgln {
    ($($arg:tt)*) => {
        if DEBUG {
            println!($($arg)*); // Allow println
        }
    }
}

struct Inliner {
    asm: Assembly,
    mapping: Vec<usize>,
}

impl Inliner {
    #[track_caller]
    fn push(&mut self, instr: Instr) {
        self.asm.instrs.push(instr)
    }
    #[track_caller]
    fn push_all(&mut self, instrs: impl IntoIterator<Item = Instr>) {
        dbgw!("  push");
        for instr in instrs {
            dbgw!(" {instr:?}");
            self.push(instr);
        }
        dbgln!();
    }
    fn push_inline(&mut self, count: usize, span: usize) {
        if count > 0 {
            self.push(Instr::push_inline(count, span));
        }
    }
    fn pop_inline(&mut self, count: usize, span: usize) {
        if count > 0 {
            self.push(Instr::pop_inline(count, span));
        }
    }
    fn copy_inline(&mut self, count: usize, span: usize) {
        if count > 0 {
            self.push(Instr::copy_inline(count, span));
        }
    }
    fn touch_stack(&mut self, count: usize, span: usize) {
        if count > 0 {
            self.push(Instr::TouchStack { count, span });
        }
    }
    fn func_instrs(&mut self, f: &Function, span: usize) {
        dbgln!(
            "push func instrs of {f} {}-{}",
            f.slice.start,
            f.slice.end()
        );
        self.push(Instr::PushFunc(f.clone()));
        self.push(Instr::Call(span));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Sequence)]
enum InlinePass {
    Inversion,
    InlineMod,
}

pub fn inline_assembly(asm: &mut Assembly) {
    for pass in all::<InlinePass>() {
        inline_pass(asm, pass);
    }
}

fn inline_pass(asm: &mut Assembly, pass: InlinePass) {
    use {ImplPrimitive::*, InlinePass::*, Instr::*, Primitive::*};
    dbgln!(
        "\ninlining {pass:?}\n{:#?}",
        crate::FmtInstrs(&asm.instrs, asm)
    );
    let mut inline_asm = asm.clone();
    inline_asm.instrs = EcoVec::new();
    let mut inliner = Inliner {
        asm: inline_asm,
        mapping: Vec::new(),
    };
    let mut i = 0;

    // // Reorganize functions
    // let mut functions = Vec::new();
    // for instr in &asm.instrs {
    //     if let Instr::PushFunc(f) = instr {
    //         functions.push(f.clone());
    //     }
    // }
    // let mut function_map: HashMap<Function, FuncSlice> = HashMap::new();
    // dbgln!("\nMap functions");
    // while !functions.is_empty() {
    //     dbgln!(" begin pass");
    //     functions.retain(|f| {
    //         let instrs = f.instrs(asm);
    //         for instr in instrs {
    //             if let Instr::PushFunc(f) = instr {
    //                 if !function_map.contains_key(f) {
    //                     return true;
    //                 }
    //             }
    //         }
    //         let start_inlined_len = inliner.asm.instrs.len();
    //         for instr in instrs {
    //             match instr {
    //                 Instr::PushFunc(f) => {
    //                     let mut f = f.clone();
    //                     f.slice = function_map[&f];
    //                     inliner.asm.instrs.push(Instr::PushFunc(f))
    //                 }
    //                 instr => inliner.asm.instrs.push(instr.clone()),
    //             }
    //         }
    //         let end_inlined_len = inliner.asm.instrs.len();
    //         let len = end_inlined_len - start_inlined_len;
    //         let slice = FuncSlice {
    //             start: start_inlined_len,
    //             len,
    //         };
    //         dbgln!(
    //             "  {f} {}-{} -> {}-{}",
    //             f.slice.start,
    //             f.slice.end(),
    //             slice.start,
    //             slice.end()
    //         );
    //         function_map.insert(f.clone(), slice);
    //         false
    //     });
    // }
    // dbgln!();

    while i < asm.instrs.len() {
        dbgln!("i: {i}");
        let start_inlined_len = inliner.asm.instrs.len();
        let new_input = match (pass, &asm.instrs[i..]) {
            (_, []) => break,
            (InlineMod, [PushFunc(f), Prim(Dip, span), inp @ ..]) => {
                inliner.push_inline(1, *span);
                inliner.func_instrs(f, *span);
                inliner.pop_inline(1, *span);
                inp
            }
            (InlineMod, [PushFunc(f), Prim(Gap, span), inp @ ..]) => {
                inliner.push(Prim(Pop, *span));
                inliner.func_instrs(f, *span);
                inp
            }
            (InlineMod, [PushFunc(f), Prim(Both, span), inp @ ..]) => {
                inliner.push_inline(f.signature().args, *span);
                inliner.func_instrs(f, *span);
                inliner.pop_inline(f.signature().args, *span);
                inliner.func_instrs(f, *span);
                inp
            }
            (InlineMod, [PushFunc(f), ImplPrim(UnBoth, span), inp @ ..]) => {
                inliner.func_instrs(f, *span);
                inliner.push_inline(f.signature().outputs, *span);
                inliner.func_instrs(f, *span);
                inliner.pop_inline(f.signature().outputs, *span);
                inp
            }
            (InlineMod, [PushFunc(g), PushFunc(f), Prim(Fork, span), inp @ ..]) => {
                inliner.copy_inline(f.signature().args, *span);
                inliner.func_instrs(g, *span);
                inliner.pop_inline(f.signature().args, *span);
                inliner.func_instrs(f, *span);
                inp
            }
            (InlineMod, [PushFunc(g), PushFunc(f), Prim(Bracket, span), inp @ ..]) => {
                inliner.push_inline(f.signature().args, *span);
                inliner.func_instrs(g, *span);
                inliner.pop_inline(f.signature().args, *span);
                inliner.func_instrs(f, *span);
                inp
            }
            (InlineMod, [PushFunc(f), Prim(On, span), inp @ ..]) => {
                if f.signature().args == 0 {
                    inliner.push_inline(1, *span);
                } else {
                    inliner.copy_inline(1, *span);
                }
                inliner.func_instrs(f, *span);
                inliner.pop_inline(1, *span);
                inp
            }
            (InlineMod, [PushFunc(f), Prim(By, span), inp @ ..]) => {
                inliner.push_inline(f.signature().args.saturating_sub(1), *span);
                if f.signature().args > 0 {
                    inliner.push(Prim(Dup, *span));
                }
                inliner.pop_inline(f.signature().args.saturating_sub(1), *span);
                inliner.func_instrs(f, *span);
                inp
            }
            (InlineMod, [PushFunc(f), Prim(With, span), inp @ ..]) => {
                let mut sig = f.signature();
                if sig.args < 2 {
                    inliner.touch_stack(2, *span);
                    sig.outputs += 2 - sig.args;
                    sig.args = 2;
                }
                inliner.push_inline(sig.args - 1, *span);
                inliner.push(Prim(Dup, *span));
                inliner.pop_inline(sig.args - 1, *span);
                inliner.func_instrs(f, *span);
                if sig.outputs >= 2 {
                    inliner.push_inline(sig.outputs - 1, *span);
                    for _ in 0..sig.outputs - 1 {
                        inliner.push(Prim(Flip, *span));
                        inliner.pop_inline(1, *span);
                    }
                }
                inliner.push(Prim(Flip, *span));
                inp
            }
            (InlineMod, [PushFunc(f), Prim(Off, span), inp @ ..]) => {
                let mut sig = f.signature();
                if sig.args < 2 {
                    inliner.touch_stack(2, *span);
                    sig.outputs += 2 - sig.args;
                    sig.args = 2;
                }
                inliner.push(Instr::Prim(Dup, *span));
                for _ in 0..sig.args - 1 {
                    inliner.push_inline(1, *span);
                    inliner.push(Instr::Prim(Flip, *span));
                }
                inliner.pop_inline(sig.args - 1, *span);
                inliner.func_instrs(f, *span);
                inp
            }
            (InlineMod, [PushFunc(f), Prim(Below, span), inp @ ..]) => {
                let mut sig = f.signature();
                if sig.args < 2 {
                    sig.args += 1;
                    sig.outputs += 1;
                }
                inliner.copy_inline(sig.args, *span);
                inliner.pop_inline(sig.args, *span);
                inliner.func_instrs(f, *span);
                inp
            }
            (InlineMod, [PushFunc(f), Prim(Above, span), inp @ ..]) => {
                let mut sig = f.signature();
                if sig.args < 2 {
                    sig.args += 1;
                    sig.outputs += 1;
                }
                inliner.copy_inline(sig.args, *span);
                inliner.func_instrs(f, *span);
                inliner.pop_inline(sig.args, *span);
                inp
            }
            (Inversion, [PushFunc(g), PushFunc(f), Prim(Under, span), ..]) => {
                let span = *span;
                let f_instrs = f.instrs(asm).to_vec();
                let g_sig = g.signature();
                let (before, after) = under_instrs(&f_instrs, g_sig, asm).unwrap();
                inliner.push_all(before);
                let [PushFunc(g), _, _, inp @ ..] = &asm.instrs[i..] else {
                    unreachable!()
                };
                inliner.func_instrs(g, span);
                inliner.push_all(after);
                inp
            }
            (_, [Comment(_), inp @ ..]) => inp,
            (_, [instr, inp @ ..]) => {
                inliner.push(instr.clone());
                inp
            }
        };
        let consumed_len = asm.instrs.len() - i - new_input.len();
        for _ in 0..consumed_len {
            inliner.mapping.push(start_inlined_len);
        }
        i += consumed_len;
    }

    // Final mapping
    inliner.mapping.push(inliner.asm.instrs.len());

    // Map unmapped functions
    dbgln!("\nmap functions:");
    for instr in inliner.asm.instrs.make_mut() {
        match instr {
            Instr::PushFunc(f) => f.slice = map_slice(&inliner.mapping, f.slice),
            Instr::CustomInverse(ci, _) => {
                if let Some(f) = ci.un.as_mut() {
                    f.slice = map_slice(&inliner.mapping, f.slice);
                }
                if let Some(f) = ci.anti.as_mut() {
                    f.slice = map_slice(&inliner.mapping, f.slice);
                }
                if let Some((b, a)) = ci.under.as_mut() {
                    b.slice = map_slice(&inliner.mapping, b.slice);
                    a.slice = map_slice(&inliner.mapping, a.slice);
                }
            }
            _ => (),
        }
    }

    // Update top slices
    inliner.asm.top_slices.clear();
    dbgln!("\nmap top slices:");
    for &top_slice in &asm.top_slices {
        (inliner.asm.top_slices).push(map_slice(&inliner.mapping, top_slice));
    }

    *asm = inliner.asm;
    dbgln!("\nmapping: {:?}", inliner.mapping);
    dbgln!("\ninlined to:\n{:?}", &asm.instrs);
    dbgln!("\ninlined to:\n{:#?}", crate::FmtInstrs(&asm.instrs, asm));
    dbgln!("top slices: {:?}", asm.top_slices);
}

fn map_slice(mapping: &[usize], slice: FuncSlice) -> FuncSlice {
    let start = mapping[slice.start];
    let end = mapping[slice.end()];
    dbgln!("map {}-{} -> {}-{}", slice.start, slice.end(), start, end);
    FuncSlice {
        start,
        len: end - start,
    }
}
