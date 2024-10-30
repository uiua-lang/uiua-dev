use crate::SigNode;

use super::*;

impl Node {
    /// Get both parts of this node's under inverse
    pub fn under_inverse(&self, g_sig: Signature, asm: &Assembly) -> InversionResult<(Node, Node)> {
        dbgln!("under-inverting {self:?}");
        under_inverse(self.as_slice(), g_sig, asm)
    }
}

impl SigNode {
    /// Get both parts of this node's under inverse
    pub fn under_inverse(
        &self,
        g_sig: Signature,
        asm: &Assembly,
    ) -> InversionResult<(SigNode, SigNode)> {
        let (before, after) = self.node.under_inverse(g_sig, asm)?;
        let (before, after) = (before.sig_node()?, after.sig_node()?);
        Ok((before, after))
    }
}

fn under_inverse(
    input: &[Node],
    g_sig: Signature,
    asm: &Assembly,
) -> InversionResult<(Node, Node)> {
    if input.is_empty() {
        return Ok((Node::empty(), Node::empty()));
    }
    let mut before = Node::empty();
    let mut after = Node::empty();
    let mut curr = input;
    let mut error = Generic;
    'find_pattern: loop {
        for pattern in UNDER_PATTERNS {
            match pattern.under_extract(curr, g_sig, asm) {
                Ok((new, bef, aft)) => {
                    dbgln!(
                        "matched pattern {pattern:?}\n  on {curr:?}\n  to  {bef:?}\n  and {aft:?}"
                    );
                    after.prepend(aft);
                    before.push(bef);
                    if new.is_empty() {
                        dbgln!("under-inverted\n  {input:?}\n  to  {before:?}\n  and {after:?}");
                        return Ok((before, after));
                    }
                    curr = new;
                    continue 'find_pattern;
                }
                Err(e) => error = error.max(e),
            }
        }
        break;
    }
    Err(error)
}

static UNDER_PATTERNS: &[&dyn UnderPattern] = &[
    &DipPat,
    &BothPat,
    &Trivial,
    &SwitchPat,
    &PartitionPat,
    &GroupPat,
    &EachPat,
    &RowsPat,
    &RepeatPat,
    &FoldPat,
    &ReversePat,
    &TransposePat,
    &RotatePat,
    &CustomPat,
    // Sign ops
    &Stash(1, Abs, (Sign, Mul)),
    &Stash(1, Sign, (Abs, Mul)),
    // Array restructuring
    &Stash(2, Take, UndoTake),
    &Stash(2, Drop, UndoDrop),
    &MaybeVal((
        Keep,
        (CopyUnd(2), Keep),
        (PopUnd(1), Flip, PopUnd(1), UndoKeep),
    )),
    &MaybeVal((
        Join,
        (Over, Shape, Over, Shape, PushUnd(2), Join),
        (PopUnd(2), UndoJoin),
    )),
    // Rise and fall
    &(
        Rise,
        (CopyUnd(1), Rise, Dup, Rise, PushUnd(1)),
        (PopUnd(1), Select, PopUnd(1), Flip, Select),
    ),
    &(
        Fall,
        (CopyUnd(1), Fall, Dup, Rise, PushUnd(1)),
        (PopUnd(1), Select, PopUnd(1), Flip, Select),
    ),
    // Sort
    &(
        Sort,
        (Dup, Rise, CopyUnd(1), Select),
        (PopUnd(1), Rise, Select),
    ),
    &(
        SortDown,
        (Dup, Fall, CopyUnd(1), Select),
        (PopUnd(1), Rise, Select),
    ),
    // Pop
    &(Pop, PushUnd(1), PopUnd(1)),
    // Value retrieval
    &Stash(1, First, UndoFirst),
    &Stash(1, Last, UndoLast),
    &Stash(2, Pick, UndoPick),
    &Stash(2, Select, UndoSelect),
    // Map control
    &MaybeVal((Get, (CopyUnd(2), Get), (PopUnd(1), Flip, PopUnd(1), Insert))),
    &Stash(2, Remove, UndoRemove),
    &MaybeVal((Insert, (CopyUnd(3), Insert), (PopUnd(3), UndoInsert))),
    // Shaping
    &(Fix, (Fix), (UndoFix)),
    &(UndoFix, (UndoFix), (Fix)),
    &Stash(1, Shape, (Flip, Reshape)),
    &(
        Len,
        (CopyUnd(1), Shape, CopyUnd(1), First),
        (PopUnd(1), UndoFirst, PopUnd(1), Flip, Reshape),
    ),
    &(
        Deshape,
        (Dup, Shape, PushUnd(1), Deshape),
        (PopUnd(1), UndoDeshape),
    ),
    &MaybeVal((
        Rerank,
        (Over, Shape, Over, PushUnd(2), Rerank),
        (PopUnd(2), UndoRerank),
    )),
    &MaybeVal((
        Reshape,
        (Over, Shape, PushUnd(1), Reshape),
        (PopUnd(1), UndoReshape),
    )),
    &MaybeVal((Chunks, (CopyUnd(1), Chunks), (PopUnd(1), UndoChunks))),
    &MaybeVal((Windows, (CopyUnd(1), Windows), (PopUnd(1), UndoWindows))),
    // Classify and deduplicate
    &(
        Classify,
        (Dup, Deduplicate, PushUnd(1), Classify),
        (PopUnd(1), Flip, Select),
    ),
    &(
        Deduplicate,
        (Dup, Classify, PushUnd(1), Deduplicate),
        (PopUnd(1), Select),
    ),
    // Where and un bits
    &(
        Where,
        (Dup, Shape, PushUnd(1), Where),
        (PopUnd(1), UndoWhere),
    ),
    &(
        UnBits,
        (Dup, Shape, PushUnd(1), UnBits),
        (PopUnd(1), UndoUnbits),
    ),
    // Rounding
    &(
        Floor,
        (Dup, Floor, Flip, Over, Sub, PushUnd(1)),
        (PopUnd(1), Add),
    ),
    &(
        Ceil,
        (Dup, Ceil, Flip, Over, Sub, PushUnd(1)),
        (PopUnd(1), Add),
    ),
    &(
        Round,
        (Dup, Round, Flip, Over, Sub, PushUnd(1)),
        (PopUnd(1), Add),
    ),
    // System stuff
    &(Now, (Now, PushUnd(1)), (Now, PopUnd(1), Sub)),
    &MaybeVal(Store1Copy(Sys(SysOp::FOpen), Sys(SysOp::Close))),
    &MaybeVal(Store1Copy(Sys(SysOp::FCreate), Sys(SysOp::Close))),
    &MaybeVal(Store1Copy(Sys(SysOp::TcpConnect), Sys(SysOp::Close))),
    &MaybeVal(Store1Copy(Sys(SysOp::TlsConnect), Sys(SysOp::Close))),
    &MaybeVal(Store1Copy(Sys(SysOp::TcpAccept), Sys(SysOp::Close))),
    &MaybeVal(Store1Copy(Sys(SysOp::TcpListen), Sys(SysOp::Close))),
    &MaybeVal(Store1Copy(Sys(SysOp::TlsListen), Sys(SysOp::Close))),
    &MaybeVal(Stash(1, Sys(SysOp::FReadAllStr), Sys(SysOp::FWriteAll))),
    &MaybeVal(Stash(1, Sys(SysOp::FReadAllBytes), Sys(SysOp::FWriteAll))),
    &MaybeVal((
        Sys(SysOp::RunStream),
        (Sys(SysOp::RunStream), CopyUnd(3)),
        (PopUnd(3), TryClose, TryClose, TryClose),
    )),
    &MaybeVal((
        Sys(SysOp::RawMode),
        (UnRawMode, PushUnd(1), Sys(SysOp::RawMode)),
        (PopUnd(1), Sys(SysOp::RawMode)),
    )),
    // Patterns that need to be last
    &StashAntiPat,
    &StashContraPat,
    &FromUnPat,
];

trait UnderPattern: fmt::Debug + Sync {
    fn under_extract<'a>(
        &self,
        input: &'a [Node],
        g_sig: Signature,
        asm: &Assembly,
    ) -> InversionResult<(&'a [Node], Node, Node)>;
}

macro_rules! under {
    // Optional parens
    ($(#[$attr:meta])* $($doc:literal,)? ($($tt:tt)*), $body:expr) => {
        under!($(#[$attr])* $($doc,)? $($tt)*, $body);
    };
    // Optional parens
    ($(#[$attr:meta])* $($doc:literal,)? ($($tt:tt)*), ref, $pat:pat, $body:expr) => {
        under!($(#[$attr])* $($doc,)? $($tt)*, ref, $pat, $body);
    };
    // Main impl
    ($(#[$attr:meta])* $($doc:literal,)? $name:ident, $input:ident, $g_sig:tt, $asm:tt, $body:expr) => {
        #[derive(Debug)]
        $(#[$attr])*
        $(#[doc = $doc])?
        struct $name;
        impl UnderPattern for $name {
            fn under_extract<'a>(
                &self,
                $input: &'a [Node],
                $g_sig: Signature,
                $asm: &Assembly,
            ) -> InversionResult<(&'a [Node], Node, Node)> {
                $body
            }
        }
    };
    // Ref pattern
    ($(#[$attr:meta])* $($doc:literal)? $name:ident, $input:ident, $g_sig:tt, $asm:tt, ref, $pat:pat, $body:expr) => {
        under!($([$attr])* $($doc)? $name, $input, $g_sig, $asm, {
            let [$pat, ref $input @ ..] = $input else {
                return generic();
            };
            $body
        });
    };
    // Non-ref pattern
    ($(#[$attr:meta])* $($doc:literal)? $name:ident, $input:ident, $g_sig:tt, $asm:tt, $pat:pat, $body:expr) => {
        under!($([$attr])* $($doc)? $name, $input, $g_sig, $asm, {
            let &[$pat, ref $input @ ..] = $input else {
                return generic();
            };
            $body
        });
    };
    // Mod pattern
    ($(#[$attr:meta])* $($doc:literal)? $name:ident, $input:ident, $g_sig:tt, $asm:tt, $prim:ident, $span:ident, $args:pat, $body:expr) => {
        under!($([$attr])* $($doc)? $name, $input, $g_sig, $asm, ref, Mod($prim, args, $span), {
            let $args = args.as_slice() else {
                return generic();
            };
            let $span = *$span;
            $body
        });
    };
}

under!(DipPat, input, g_sig, asm, Dip, span, [f], {
    // F inverse
    let inner_g_sig = Signature::new(g_sig.args.saturating_sub(1), g_sig.outputs);
    let (f_before, f_after) = f.under_inverse(inner_g_sig, asm)?;
    // Rest inverse
    let (rest_before, rest_after) = under_inverse(input, g_sig, asm)?;
    let rest_before_sig = rest_before.sig()?;
    let rest_after_sig = rest_after.sig()?;
    // Make before
    let mut before = Mod(Dip, eco_vec![f_before], span);
    before.push(rest_before);
    // Make after
    let mut after = rest_after;
    let after_inner = if g_sig.args + rest_before_sig.args <= g_sig.outputs + rest_after_sig.outputs
    {
        Mod(Dip, eco_vec![f_after], span)
    } else {
        f_after.node
    };
    after.push(after_inner);
    Ok((&[], before, after))
});

under!(BothPat, input, g_sig, asm, Both, span, [f], {
    let (f_before, f_after) = if g_sig.args > g_sig.outputs {
        let inv = f.un_inverse(asm)?;
        (f.clone(), inv)
    } else {
        let inner_g_sig = Signature::new(g_sig.args.saturating_sub(1), g_sig.outputs);
        f.under_inverse(inner_g_sig, asm)?
    };
    let before = Mod(Both, eco_vec![f_before], span);
    let after = if g_sig.args > g_sig.outputs {
        f_after.node
    } else {
        ImplMod(UnBoth, eco_vec![f_after], span)
    };
    Ok((input, before, after))
});

under!(
    "Derives under inverses from un inverses",
    (FromUnPat, input, _, asm),
    {
        for pat in UN_PATTERNS {
            if let Ok((new, inv)) = pat.invert_extract(input, asm) {
                let nodes = &input[..input.len() - new.len()];
                return Ok((new, Node::from(nodes), inv));
            }
        }
        generic()
    }
);

under!(
    "Derives under inverses from anti inverses",
    (StashAntiPat, input, _, asm),
    {
        for pat in ANTI_PATTERNS {
            if let Ok((new, inv)) = pat.invert_extract(input, asm) {
                let nodes = &input[..input.len() - new.len()];
                let span = nodes
                    .iter()
                    .find_map(Node::span)
                    .or_else(|| inv.span())
                    .unwrap_or(0);
                let before = Node::from_iter([CopyToUnder(1, span), Node::from(nodes)]);
                let after = Node::from_iter([PopUnder(1, span), inv]);
                return Ok((new, before, after));
            }
        }
        generic()
    }
);

under!(
    "Derives under inverses from contra inverses",
    (StashContraPat, input, _, asm),
    {
        for pat in CONTRA_PATTERNS {
            if let Ok((new, inv)) = pat.invert_extract(input, asm) {
                let nodes = &input[..input.len() - new.len()];
                let span = nodes
                    .iter()
                    .find_map(Node::span)
                    .or_else(|| inv.span())
                    .unwrap_or(0);
                let before =
                    Node::from_iter([Prim(Over, span), PushUnder(1, span), Node::from(nodes)]);
                let after = Node::from_iter([PopUnder(1, span), Prim(Flip, span), inv]);
                return Ok((new, before, after));
            }
        }
        generic()
    }
);

under!(EachPat, input, g_sig, asm, Each, span, [f], {
    let (f_before, f_after) = f.under_inverse(g_sig, asm)?;
    let befores = Mod(Each, eco_vec![f_before], span);
    let afters = Mod(Each, eco_vec![f_after], span);
    Ok((input, befores, afters))
});

under!(RowsPat, input, g_sig, asm, {
    let [Mod(prim @ (Rows | Inventory), args, span), input @ ..] = input else {
        return generic();
    };
    let [f] = args.as_slice() else {
        return generic();
    };
    let (f_before, f_after) = f.under_inverse(g_sig, asm)?;
    let befores = Mod(*prim, eco_vec![f_before], *span);
    let after_sig = f_after.sig;
    let mut afters = Node::from_iter([Prim(Reverse, *span), Mod(*prim, eco_vec![f_after], *span)]);
    afters.push(ImplPrim(
        UndoReverse {
            n: after_sig.outputs,
            all: true,
        },
        *span,
    ));
    Ok((input, befores, afters))
});

under!(RepeatPat, input, g_sig, asm, {
    let (f, span, input) = match input {
        [Mod(Repeat, args, span), input @ ..] => {
            let [f] = args.as_slice() else {
                return generic();
            };
            (f, *span, input)
        }
        [ImplMod(RepeatWithInverse, args, span), input @ ..] => {
            let [f, _] = args.as_slice() else {
                return generic();
            };
            (f, *span, input)
        }
        _ => return generic(),
    };
    let (f_before, f_after) = f.under_inverse(g_sig, asm)?;
    let befores = Node::from_iter([CopyToUnder(1, span), Mod(Repeat, eco_vec![f_before], span)]);
    let afters = Node::from_iter([PopUnder(1, span), Mod(Repeat, eco_vec![f_after], span)]);
    Ok((input, befores, afters))
});

under!(FoldPat, input, g_sig, asm, Fold, span, [f], {
    let (f_before, f_after) = f.under_inverse(g_sig, asm)?;
    if f_before.sig.outputs > f_before.sig.args || f_after.sig.outputs > f_after.sig.args {
        return generic();
    }
    let befores = Node::from_iter([
        Prim(Dup, span),
        Prim(Len, span),
        PushUnder(1, span),
        Mod(Fold, eco_vec![f_before], span),
    ]);
    let afters = Node::from_iter([PopUnder(1, span), Mod(Repeat, eco_vec![f_after], span)]);
    Ok((input, befores, afters))
});

under!(CustomPat, input, _, _, ref, CustomInverse(cust, span), {
    let normal = cust.normal.clone()?;
    let (mut before, mut after, to_save) = if let Some((before, after)) = cust.under.clone() {
        if before.sig.outputs < normal.sig.outputs {
            return generic();
        }
        let to_save = before.sig.outputs - normal.sig.outputs;
        (before.node, after.node, to_save)
    } else if let Some(anti) = cust.anti.clone() {
        let to_save = anti.sig.args - normal.sig.outputs;
        let before = Mod(On, eco_vec![normal.clone()], *span);
        let after = anti.node;
        (before, after, to_save)
    } else {
        return generic();
    };
    if to_save > 0 {
        before.push(PushUnder(to_save, *span));
        after.prepend(PopUnder(to_save, *span));
    }
    Ok((input, before, after))
});

#[derive(Debug)]
struct Trivial;
impl UnderPattern for Trivial {
    fn under_extract<'a>(
        &self,
        input: &'a [Node],
        g_sig: Signature,
        asm: &Assembly,
    ) -> InversionResult<(&'a [Node], Node, Node)> {
        match input {
            [NoInline(inner), input @ ..] => {
                let (before, after) = inner.under_inverse(g_sig, asm)?;
                Ok((input, NoInline(before.into()), NoInline(after.into())))
            }
            [TrackCaller(inner), input @ ..] => {
                let (before, after) = inner.under_inverse(g_sig, asm)?;
                Ok((input, TrackCaller(before.into()), TrackCaller(after.into())))
            }
            [node @ SetOutputComment { .. }, input @ ..] => {
                Ok((input, node.clone(), Node::empty()))
            }
            [Call(f, _), input @ ..] => {
                let (before, after) = asm[f].under_inverse(g_sig, asm).map_err(|e| e.func(f))?;
                Ok((input, before, after))
            }
            _ => generic(),
        }
    }
}

under!(
    (SwitchPat, input, g_sig, asm),
    ref,
    Node::Switch {
        branches,
        sig,
        span,
        under_cond: false
    },
    {
        let mut befores = EcoVec::with_capacity(branches.len());
        let mut afters = EcoVec::with_capacity(branches.len());
        let mut undo_sig: Option<Signature> = None;
        for branch in branches {
            // Calc under f
            let (before, after) = branch.under_inverse(g_sig, asm)?;
            let after_sig = after.sig;
            befores.push(before);
            afters.push(after);
            // Aggregate sigs
            let undo_sig = undo_sig.get_or_insert(after_sig);
            if after_sig.is_compatible_with(*undo_sig) {
                *undo_sig = undo_sig.max_with(after_sig);
            } else if after_sig.outputs == undo_sig.outputs {
                undo_sig.args = undo_sig.args.max(after_sig.args)
            } else {
                return generic();
            }
        }
        let before = Node::Switch {
            branches: befores,
            sig: *sig,
            span: *span,
            under_cond: true,
        };
        let after = Node::from_iter([
            Node::PopUnder(1, *span),
            Node::Switch {
                branches: afters,
                sig: undo_sig.ok_or(Generic)?,
                span: *span,
                under_cond: false,
            },
        ]);
        Ok((input, before, after))
    }
);

macro_rules! partition_group {
    ($name:ident, $prim:ident, $impl_prim1:ident, $impl_prim2:ident) => {
        under!($name, input, g_sig, asm, $prim, span, [f], {
            let (f_before, f_after) = f.under_inverse(g_sig, asm)?;
            let before =
                Node::from_iter([CopyToUnder(2, span), Mod($prim, eco_vec![f_before], span)]);
            let after = Node::from_iter([
                ImplMod($impl_prim1, eco_vec![f_after], span),
                Mod(Dip, eco_vec![PopUnder(2, span).sig_node()?], span),
                ImplPrim(ImplPrimitive::$impl_prim2, span),
            ]);
            Ok((input, before, after))
        });
    };
}

partition_group!(PartitionPat, Partition, UndoPartition1, UndoPartition2);
partition_group!(GroupPat, Group, UndoGroup1, UndoGroup2);

under!(ReversePat, input, g_sig, _, Prim(Reverse, span), {
    if g_sig.outputs == 1 {
        return generic();
    }
    let count = if g_sig.args == 1 || g_sig.outputs == g_sig.args * 2 {
        g_sig.outputs.max(1)
    } else {
        1
    };
    let after = ImplPrim(
        UndoReverse {
            n: count,
            all: false,
        },
        span,
    );
    Ok((input, Prim(Reverse, span), after))
});

under!(TransposePat, input, g_sig, _, {
    if g_sig.outputs == 1 {
        return generic();
    }
    let (before, span, amnt, input) = match input {
        [node @ Prim(Transpose, span), input @ ..] => (node, *span, 1, input),
        [node @ ImplPrim(TransposeN(amnt), span), input @ ..] => (node, *span, *amnt, input),
        _ => return generic(),
    };
    let count = if g_sig.args == 1 || g_sig.outputs == g_sig.args * 2 {
        g_sig.outputs.max(1)
    } else {
        1
    };
    let after = ImplPrim(UndoTransposeN(count, amnt), span);
    Ok((input, before.clone(), after))
});

under!(RotatePat, input, g_sig, _, Prim(Rotate, span), {
    let count = if g_sig.args == 1 || g_sig.outputs == g_sig.args * 2 {
        g_sig.outputs.max(1)
    } else {
        1
    };
    let before = Node::from_iter([CopyToUnder(1, span), Prim(Rotate, span)]);
    let after = Node::from_iter([PopUnder(1, span), ImplPrim(UndoRotate(count), span)]);
    Ok((input, before, after))
});

/// Copy some values to the under stack at the beginning of the "do" step
/// and pop them at the beginning of the "undo" step
///
/// Allows a leading value if staching at least 2 values
#[derive(Debug)]
struct Stash<A, B>(usize, A, B);
impl<A, B> UnderPattern for Stash<A, B>
where
    A: SpanFromNodes + AsNode + Copy,
    B: AsNode + Copy,
{
    fn under_extract<'a>(
        &self,
        input: &'a [Node],
        g_sig: Signature,
        asm: &Assembly,
    ) -> InversionResult<(&'a [Node], Node, Node)> {
        let &Stash(n, a, b) = self;
        let pat = (a, (CopyUnd(n), a), (PopUnd(n), b));
        if n >= 2 {
            MaybeVal(pat).under_extract(input, g_sig, asm)
        } else {
            pat.under_extract(input, g_sig, asm)
        }
    }
}
impl<P: UnderPattern> UnderPattern for MaybeVal<P> {
    fn under_extract<'a>(
        &self,
        mut input: &'a [Node],
        g_sig: Signature,
        asm: &Assembly,
    ) -> InversionResult<(&'a [Node], Node, Node)> {
        let val = if let Ok((inp, val)) = Val.invert_extract(input, asm) {
            input = inp;
            Some(val)
        } else {
            None
        };
        let MaybeVal(p) = self;
        let (input, mut before, after) = p.under_extract(input, g_sig, asm)?;
        if let Some(val) = val {
            before.prepend(val);
        }
        Ok((input, before, after))
    }
}

#[derive(Debug)]
struct Store1Copy<A, B>(A, B);
impl<A, B> UnderPattern for Store1Copy<A, B>
where
    A: SpanFromNodes + AsNode + Copy,
    B: AsNode + Copy,
{
    fn under_extract<'a>(
        &self,
        input: &'a [Node],
        g_sig: Signature,
        asm: &Assembly,
    ) -> InversionResult<(&'a [Node], Node, Node)> {
        let &Store1Copy(a, b) = self;
        MaybeVal((a, (a, CopyUnd(1)), (PopUnd(1), b))).under_extract(input, g_sig, asm)
    }
}

impl<A, B, C> UnderPattern for (A, B, C)
where
    A: SpanFromNodes,
    B: AsNode,
    C: AsNode,
{
    fn under_extract<'a>(
        &self,
        input: &'a [Node],
        _: Signature,
        asm: &Assembly,
    ) -> InversionResult<(&'a [Node], Node, Node)> {
        let (a, b, c) = self;
        let (input, span) = a.span_from_nodes(input, asm).ok_or(Generic)?;
        let span = span.ok_or(Generic)?;
        Ok((input, b.as_node(span), c.as_node(span)))
    }
}

#[derive(Debug)]
struct PushUnd(usize);
impl AsNode for PushUnd {
    fn as_node(&self, span: usize) -> Node {
        Node::PushUnder(self.0, span)
    }
}

#[derive(Debug)]
struct CopyUnd(usize);
impl AsNode for CopyUnd {
    fn as_node(&self, span: usize) -> Node {
        Node::CopyToUnder(self.0, span)
    }
}

#[derive(Debug)]
struct PopUnd(usize);
impl AsNode for PopUnd {
    fn as_node(&self, span: usize) -> Node {
        Node::PopUnder(self.0, span)
    }
}
