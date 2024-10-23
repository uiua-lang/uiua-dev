//! Logic for inlining code before running

use ecow::EcoVec;

use crate::{Assembly, FuncSlice, Function, Instr, Primitive};

#[derive(Default)]
struct Inliner {
    inlined: EcoVec<Instr>,
    mapping: Vec<usize>,
}

impl Inliner {
    fn push(&mut self, instr: Instr) {
        self.inlined.push(instr);
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
    fn map_slice(&self, slice: FuncSlice) -> FuncSlice {
        let start = self.mapping[slice.start];
        let end = self.mapping[slice.end()];
        FuncSlice {
            start,
            len: end - start,
        }
    }
    fn func_instrs(&mut self, f: &Function, span: usize) {
        if f.flags.track_caller() || f.flags.no_inline() {
            let mut f = f.clone();
            f.slice = self.map_slice(f.slice);
            self.push(Instr::PushFunc(f));
            self.push(Instr::Call(span));
        } else {
            let start = self.mapping[f.slice.start];
            let end = self.mapping[f.slice.end()];
            for i in start..end {
                self.push(self.inlined[i].clone());
            }
        }
    }
}

pub fn inline_assembly(asm: &mut Assembly) {
    // println!("inlining {}", crate::FmtInstrs(&asm.instrs, asm));
    use Instr::*;
    use Primitive::*;
    let mut input = asm.instrs.as_slice();
    let mut inliner = Inliner::default();
    loop {
        let start_inlined_len = inliner.inlined.len();
        let new_input = match input {
            [] => break,
            [PushFunc(f), Prim(Dip, span), inp @ ..] => {
                inliner.push_inline(1, *span);
                inliner.func_instrs(f, *span);
                inliner.pop_inline(1, *span);
                inp
            }
            [PushFunc(f), Prim(Gap, span), inp @ ..] => {
                inliner.push(Prim(Pop, *span));
                inliner.func_instrs(f, *span);
                inp
            }
            [PushFunc(f), Prim(Both, span), inp @ ..] => {
                inliner.push_inline(f.signature().args, *span);
                inliner.func_instrs(f, *span);
                inliner.pop_inline(f.signature().args, *span);
                inliner.func_instrs(f, *span);
                inp
            }
            [PushFunc(g), PushFunc(f), Prim(Fork, span), inp @ ..] => {
                inliner.copy_inline(f.signature().args, *span);
                inliner.func_instrs(g, *span);
                inliner.pop_inline(f.signature().args, *span);
                inliner.func_instrs(f, *span);
                inp
            }
            [PushFunc(g), PushFunc(f), Prim(Bracket, span), inp @ ..] => {
                inliner.push_inline(f.signature().args, *span);
                inliner.func_instrs(g, *span);
                inliner.pop_inline(f.signature().args, *span);
                inliner.func_instrs(f, *span);
                inp
            }
            [PushFunc(f), Prim(On, span), inp @ ..] => {
                if f.signature().args == 0 {
                    inliner.push_inline(1, *span);
                } else {
                    inliner.copy_inline(1, *span);
                }
                inliner.func_instrs(f, *span);
                inliner.pop_inline(1, *span);
                inp
            }
            [PushFunc(f), Prim(By, span), inp @ ..] => {
                inliner.push_inline(f.signature().args.saturating_sub(1), *span);
                if f.signature().args > 0 {
                    inliner.push(Prim(Dup, *span));
                }
                inliner.pop_inline(f.signature().args.saturating_sub(1), *span);
                inliner.func_instrs(f, *span);
                inp
            }
            [PushFunc(f), Prim(With, span), inp @ ..] => {
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
            [PushFunc(f), Prim(Off, span), inp @ ..] => {
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
            [PushFunc(f), Prim(Below, span), inp @ ..] => {
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
            [PushFunc(f), Prim(Above, span), inp @ ..] => {
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
            [Comment(_), inp @ ..] => inp,
            [PushFunc(f), inp @ ..] => {
                let mut f = f.clone();
                f.slice = inliner.map_slice(f.slice);
                inliner.push(Instr::PushFunc(f));
                inp
            }
            [CustomInverse(ci, span), inp @ ..] => {
                let mut ci = ci.clone();
                ci.un = ci.un.map(|mut f| {
                    f.slice = inliner.map_slice(f.slice);
                    f
                });
                ci.anti = ci.anti.map(|mut f| {
                    f.slice = inliner.map_slice(f.slice);
                    f
                });
                ci.under = ci.under.map(|mut f| {
                    f.0.slice = inliner.map_slice(f.0.slice);
                    f.1.slice = inliner.map_slice(f.1.slice);
                    f
                });
                inliner.push(Instr::CustomInverse(ci, *span));
                inp
            }
            [instr, inp @ ..] => {
                inliner.push(instr.clone());
                inp
            }
        };
        let consumed_len = input.len() - new_input.len();
        for _ in 0..consumed_len {
            inliner.mapping.push(start_inlined_len);
        }
        input = new_input;
    }

    inliner.mapping.push(inliner.inlined.len());

    // Update top slices
    for top_slice in &mut asm.top_slices {
        *top_slice = inliner.map_slice(*top_slice);
    }

    // println!("inlined to {}", crate::FmtInstrs(&inliner.inlined, asm));
    asm.instrs = inliner.inlined;
}
