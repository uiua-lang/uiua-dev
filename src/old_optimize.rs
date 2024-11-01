use std::fmt;

use ecow::EcoVec;

use crate::{check::instrs_clean_signature, Assembly, ImplPrimitive, Instr, Primitive, TempStack};

pub(crate) fn optimize_instrs_mut(
    instrs: &mut EcoVec<Instr>,
    mut new: Instr,
    maximal: bool,
    asm: &Assembly,
) {
    use ImplPrimitive::*;
    use Primitive::*;
    if let Instr::Push(val) = &mut new {
        val.compress();
    }
    match (instrs.make_mut(), new) {
        // Dips
        (
            [.., Instr::PushTemp {
                stack: a_stack,
                count: a_count,
                ..
            }],
            Instr::PopTemp {
                stack: b_stack,
                count: b_count,
                span,
                ..
            },
        ) if maximal && *a_stack == b_stack && *a_count == b_count => {
            let count = *a_count;
            instrs.pop();
            instrs.push(Instr::TouchStack { count, span });
        }
        (
            [.., Instr::PushTemp {
                stack: a_stack,
                count: a_count,
                ..
            }, Instr::Prim(Identity, span)],
            Instr::PopTemp {
                stack: b_stack,
                count: b_count,
                ..
            },
        ) if maximal && *a_stack == b_stack && *a_count == b_count => {
            let span = *span;
            let count = *a_count + 1;
            instrs.pop();
            instrs.pop();
            instrs.push(Instr::TouchStack { count, span });
        }
        (
            [.., Instr::PushTemp {
                stack: a_stack,
                count: a_count,
                ..
            }, Instr::TouchStack { count, span }],
            Instr::PopTemp {
                stack: b_stack,
                count: b_count,
                ..
            },
        ) if maximal && *a_stack == b_stack && *a_count == b_count => {
            let span = *span;
            let count = *a_count + *count;
            instrs.pop();
            instrs.pop();
            instrs.push(Instr::TouchStack { count, span });
        }
        // By dup
        (
            &mut [.., Instr::PushTemp {
                stack: TempStack::Inline,
                count: 1,
                span: push_temp_span,
            }, Instr::Prim(Dup, dup_span)],
            Instr::PopTemp {
                stack: TempStack::Inline,
                count: 1,
                span: pop_temp_span,
            },
        ) => {
            let end = instrs.len() - 2;
            let pop_temp = Instr::PopTemp {
                stack: TempStack::Inline,
                count: 1,
                span: pop_temp_span,
            };
            if let Some(i) = (0..end)
                .find(|&i| instrs_clean_signature(&instrs[i..end]).is_some_and(|sig| sig == (1, 1)))
            {
                instrs.truncate(end);
                let one_one = instrs[i..].to_vec();
                instrs.truncate(i);
                instrs.push(Instr::PushTemp {
                    stack: TempStack::Inline,
                    count: 1,
                    span: push_temp_span,
                });
                instrs.push(Instr::Prim(Dup, dup_span));
                instrs.push(pop_temp);
                instrs.extend(one_one);
            } else {
                instrs.push(pop_temp);
            }
        }
        ([.., Instr::TouchStack { count: 1, span }], Instr::Prim(Rand, _)) => {
            // By rand
            let span = *span;
            instrs.pop();
            instrs.push(Instr::Prim(Dup, span));
            instrs.push(Instr::ImplPrim(ReplaceRand, span));
        }
        (
            [.., Instr::PushTemp {
                stack: TempStack::Inline,
                count: 1,
                span,
            }, Instr::Prim(Rand, _)],
            Instr::PopTemp {
                stack: TempStack::Inline,
                count: 1,
                span: pop_span,
            },
        ) => {
            // On rand
            let span = *span;
            instrs.pop();
            instrs.pop();
            instrs.push(Instr::copy_inline(span));
            instrs.push(Instr::ImplPrim(ReplaceRand, span));
            instrs.push(Instr::pop_inline(1, pop_span));
        }
        ([.., Instr::Prim(Windows, _), Instr::PushFunc(f)], instr @ Instr::Prim(Rows, span)) => {
            match f.instrs(asm) {
                // Adjacent
                [inner @ Instr::PushFunc(reduced_f), Instr::Prim(Reduce, span)]
                    if reduced_f.signature() == (2, 1) =>
                {
                    let inner = inner.clone();
                    instrs.pop();
                    instrs.pop();
                    instrs.push(inner);
                    instrs.push(Instr::ImplPrim(ImplPrimitive::Adjacent, *span));
                }
                // Rows Windows
                _ if f.signature() == (1, 1) => {
                    let f = f.clone();
                    instrs.pop();
                    instrs.pop();
                    instrs.push(Instr::PushFunc(f));
                    instrs.push(Instr::ImplPrim(ImplPrimitive::RowsWindows, span));
                }
                _ => {
                    instrs.push(instr);
                }
            }
        }
        // Validate type
        (
            [.., Instr::Prim(Dup, _), Instr::Prim(Type, _), Instr::Push(val)],
            Instr::ImplPrim(MatchPattern, span),
        ) => {
            let val = val.clone();
            instrs.pop();
            instrs.pop();
            instrs.pop();
            instrs.push(Instr::Push(val));
            instrs.push(Instr::ImplPrim(ValidateType, span));
        }
        ([.., Instr::Prim(Type, _), Instr::Push(val)], Instr::ImplPrim(MatchPattern, span)) => {
            let val = val.clone();
            instrs.pop();
            instrs.pop();
            instrs.push(Instr::Push(val));
            instrs.push(Instr::ImplPrim(ValidateTypeConsume, span));
        }
        (
            [.., Instr::ImplPrim(
                TraceN {
                    n: a,
                    inverse: inv_a,
                    ..
                },
                _,
            )],
            Instr::ImplPrim(
                TraceN {
                    n: b,
                    inverse: inv_b,
                    ..
                },
                _,
            ),
        ) if *inv_a == inv_b => {
            *a += b;
            if *a == 0 {
                instrs.pop();
            }
        }
        (_, instr) => instrs.push(instr),
    }
}

pub(crate) fn optimize_instrs<I>(instrs: I, maximal: bool, asm: &Assembly) -> EcoVec<Instr>
where
    I: IntoIterator<Item = Instr> + fmt::Debug,
    I::IntoIter: ExactSizeIterator,
{
    // println!("optimize {:?}", instrs);
    let instrs = instrs.into_iter();
    let mut new = EcoVec::with_capacity(instrs.len());
    for instr in instrs {
        optimize_instrs_mut(&mut new, instr, maximal, asm);
    }
    // println!("to       {:?}", new);
    new
}
