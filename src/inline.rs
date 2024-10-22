//! Logic for inlining code before running

use ecow::EcoVec;

use crate::{Assembly, Function, Instr, Primitive};

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
    fn func_instrs(&mut self, f: &Function, span: usize) {
        if f.flags.track_caller() || f.flags.no_inline() {
            let mut f = f.clone();
            let end = f.slice.end();
            f.slice.start = self.mapping[f.slice.start];
            f.slice.len = self.mapping[end] - f.slice.start;
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

    let Inliner {
        inlined,
        mut mapping,
    } = inliner;
    mapping.push(inlined.len());

    // Update top slices
    for top_slice in &mut asm.top_slices {
        let start = mapping[top_slice.start];
        let end = mapping[top_slice.end()];
        let len = end - start;
        top_slice.start = start;
        top_slice.len = len;
    }
    asm.instrs = inlined;
}
