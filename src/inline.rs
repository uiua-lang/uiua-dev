//! Logic for inlining code before running

use ecow::EcoVec;

use crate::{Assembly, Function, Instr, Primitive};

pub fn inline_assembly(asm: &mut Assembly) {
    use Instr::*;
    use Primitive::*;
    let input = asm.instrs.clone();
    let mut input = input.as_slice();
    let mut inlined = EcoVec::new();
    let mut mapping = Vec::new();
    loop {
        let start_inlined_len = inlined.len();
        let new_input = match input {
            [] => break,
            [PushFunc(f), Prim(Dip, span), inp @ ..] => {
                push_inline(&mut inlined, 1, *span);
                push_func_instrs(&mut inlined, f, asm);
                pop_inline(&mut inlined, 1, *span);
                inp
            }
            [PushFunc(f), Prim(Gap, span), inp @ ..] => {
                inlined.push(Prim(Pop, *span));
                push_func_instrs(&mut inlined, f, asm);
                inp
            }
            [PushFunc(f), Prim(Both, span), inp @ ..] => {
                push_inline(&mut inlined, f.signature().args, *span);
                push_func_instrs(&mut inlined, f, asm);
                pop_inline(&mut inlined, f.signature().args, *span);
                push_func_instrs(&mut inlined, f, asm);
                inp
            }
            [PushFunc(g), PushFunc(f), Prim(Fork, span), inp @ ..] => {
                copy_inline(&mut inlined, f.signature().args, *span);
                push_func_instrs(&mut inlined, g, asm);
                pop_inline(&mut inlined, f.signature().args, *span);
                push_func_instrs(&mut inlined, f, asm);
                inp
            }
            [PushFunc(g), PushFunc(f), Prim(Bracket, span), inp @ ..] => {
                push_inline(&mut inlined, f.signature().args, *span);
                push_func_instrs(&mut inlined, g, asm);
                pop_inline(&mut inlined, f.signature().args, *span);
                push_func_instrs(&mut inlined, f, asm);
                inp
            }
            [PushFunc(f), Prim(On, span), inp @ ..] => {
                if f.signature().args == 0 {
                    push_inline(&mut inlined, 1, *span);
                } else {
                    copy_inline(&mut inlined, 1, *span);
                }
                push_func_instrs(&mut inlined, f, asm);
                pop_inline(&mut inlined, 1, *span);
                inp
            }
            [PushFunc(f), Prim(By, span), inp @ ..] => {
                push_inline(&mut inlined, f.signature().args.saturating_sub(1), *span);
                if f.signature().args > 0 {
                    inlined.push(Prim(Dup, *span));
                }
                pop_inline(&mut inlined, f.signature().args.saturating_sub(1), *span);
                push_func_instrs(&mut inlined, f, asm);
                inp
            }
            [PushFunc(f), Prim(With, span), inp @ ..] => {
                let mut sig = f.signature();
                if sig.args < 2 {
                    inlined.push(Instr::TouchStack {
                        count: 2,
                        span: *span,
                    });
                    sig.outputs += 2 - sig.args;
                    sig.args = 2;
                }
                push_inline(&mut inlined, sig.args - 1, *span);
                inlined.push(Prim(Dup, *span));
                pop_inline(&mut inlined, sig.args - 1, *span);
                push_func_instrs(&mut inlined, f, asm);
                if sig.outputs >= 2 {
                    push_inline(&mut inlined, sig.outputs - 1, *span);
                    for _ in 0..sig.outputs - 1 {
                        inlined.push(Prim(Flip, *span));
                        pop_inline(&mut inlined, 1, *span);
                    }
                }
                inlined.push(Prim(Flip, *span));
                inp
            }
            [PushFunc(f), Prim(Off, span), inp @ ..] => {
                let mut sig = f.signature();
                if sig.args < 2 {
                    inlined.push(Instr::TouchStack {
                        count: 2,
                        span: *span,
                    });
                    sig.outputs += 2 - sig.args;
                    sig.args = 2;
                }
                inlined.push(Instr::Prim(Dup, *span));
                for _ in 0..sig.args - 1 {
                    push_inline(&mut inlined, 1, *span);
                    inlined.push(Instr::Prim(Flip, *span));
                }
                pop_inline(&mut inlined, sig.args - 1, *span);
                push_func_instrs(&mut inlined, f, asm);
                inp
            }
            [PushFunc(f), Prim(Below, span), inp @ ..] => {
                let mut sig = f.signature();
                if sig.args < 2 {
                    sig.args += 1;
                    sig.outputs += 1;
                }
                copy_inline(&mut inlined, sig.args, *span);
                pop_inline(&mut inlined, sig.args, *span);
                push_func_instrs(&mut inlined, f, asm);
                inp
            }
            [PushFunc(f), Prim(Above, span), inp @ ..] => {
                let mut sig = f.signature();
                if sig.args < 2 {
                    sig.args += 1;
                    sig.outputs += 1;
                }
                copy_inline(&mut inlined, sig.args, *span);
                push_func_instrs(&mut inlined, f, asm);
                pop_inline(&mut inlined, sig.args, *span);
                inp
            }
            [Comment(_), inp @ ..] => inp,
            [instr, inp @ ..] => {
                inlined.push(instr.clone());
                inp
            }
        };
        let consumed_len = input.len() - new_input.len();
        for _ in 0..consumed_len {
            mapping.push(start_inlined_len);
        }
        input = new_input;
    }
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

fn push_func_instrs(inlined: &mut EcoVec<Instr>, func: &Function, asm: &Assembly) {
    if func.flags.track_caller() || func.flags.no_inline() {
        todo!()
    } else {
        inlined.extend(func.instrs(asm).iter().cloned());
    }
}

fn push_inline(inlined: &mut EcoVec<Instr>, count: usize, span: usize) {
    if count > 0 {
        inlined.push(Instr::push_inline(count, span));
    }
}

fn pop_inline(inlined: &mut EcoVec<Instr>, count: usize, span: usize) {
    if count > 0 {
        inlined.push(Instr::pop_inline(count, span));
    }
}

fn copy_inline(inlined: &mut EcoVec<Instr>, count: usize, span: usize) {
    if count > 0 {
        inlined.push(Instr::copy_inline(count, span));
    }
}
