use super::*;

impl Node {
    /// Get both parts of this node's under inverse
    pub fn under_inverse(&self, g_sig: Signature, asm: &Assembly) -> InversionResult<(Node, Node)> {
        dbgln!("under-inverting {self:?}");
        under_inverse(self.as_slice(), g_sig, asm)
    }
}

fn under_inverse(
    input: &[Node],
    g_sig: Signature,
    asm: &Assembly,
) -> InversionResult<(Node, Node)> {
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
    ($(#[$attr:meta])* $($doc:literal)? $name:ident, $input:ident, $g_sig:tt, $asm:tt, $body:expr) => {
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
    ($(#[$attr:meta])* $($doc:literal)? $name:ident, $input:ident, $g_sig:tt, $asm:tt, ref, $pat:pat, $body:expr) => {
        under!($([$attr])* $($doc)? $name, $input, $g_sig, $asm, {
            let [$pat, ref $input @ ..] = $input else {
                return generic();
            };
            $body
        });
    };
    ($(#[$attr:meta])* $($doc:literal)? $name:ident, $input:ident, $g_sig:tt, $asm:tt, $pat:pat, $body:expr) => {
        under!($([$attr])* $($doc)? $name, $input, $g_sig, $asm, {
            let &[$pat, ref $input @ ..] = $input else {
                return generic();
            };
            $body
        });
    };
    ($(#[$attr:meta])* $($doc:literal)? $name:ident, $input:ident, $asm:tt, $g_sig:tt, $prim:ident, $span:ident, $args:pat, $body:expr) => {
        under!($([$attr])* $($doc)? $name, $input, $g_sig, $asm, ref, Mod($prim, args, $span), {
            let $args = args.as_slice() else {
                return generic();
            };
            let $span = *$span;
            $body
        });
    };
}

under!(
    /// Derives under inverses from un inverses
    FromUnPat, input, _, asm, {
    for pat in UN_PATTERNS {
        if let Ok((new, inv)) = pat.invert_extract(input, asm) {
            let nodes = &input[..input.len() - new.len()];
            return Ok((new, Node::from(nodes), inv));
        }
    }
    generic()
});

under!(
    /// Derives under inverses from anti inverses
    StashAntiPat, input, _, asm, {
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
});

under!(
    /// Derives under inverses from contra inverses
    StashContraPat, input, _, asm, {
    for pat in CONTRA_PATTERNS {
        if let Ok((new, inv)) = pat.invert_extract(input, asm) {
            let nodes = &input[..input.len() - new.len()];
            let span = nodes
                .iter()
                .find_map(Node::span)
                .or_else(|| inv.span())
                .unwrap_or(0);
            let before = Node::from_iter([Prim(Over, span), PushUnder(1, span), Node::from(nodes)]);
            let after = Node::from_iter([PopUnder(1, span), Prim(Flip, span), inv]);
            return Ok((new, before, after));
        }
    }
    generic()
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

/// Optionally allow a leading value
///
/// The value will not be pushed during the "undo" step
#[derive(Debug)]
struct MaybeVal<P>(P);
impl<P> UnderPattern for MaybeVal<P>
where
    P: UnderPattern,
{
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
