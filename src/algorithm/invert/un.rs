use ecow::eco_vec;

use crate::SigNode;

use super::*;

impl Node {
    /// Get the un inverse of this node
    pub fn un_inverse(&self, asm: &Assembly) -> InversionResult<Node> {
        dbgln!("un-inverting {self:?}");
        un_inverse(self.as_slice(), asm)
    }
    /// Get the anti inverse of this node
    pub fn anti_inverse(&self, asm: &Assembly) -> InversionResult<Node> {
        dbgln!("anti-inverting {self:?}");
        anti_inverse(self.as_slice(), asm)
    }
}

impl SigNode {
    /// Get the un-inverse of this node
    pub fn un_inverse(&self, asm: &Assembly) -> InversionResult<SigNode> {
        let inv = self.node.un_inverse(asm)?;
        Ok(SigNode::new(inv, self.sig.inverse()))
    }
    /// Get the anti inverse of this node
    pub fn anti_inverse(&self, asm: &Assembly) -> InversionResult<SigNode> {
        let inv = self.node.anti_inverse(asm)?;
        let sig = self.sig.anti().ok_or(Generic)?;
        Ok(SigNode::new(inv, sig))
    }
}

fn un_inverse(input: &[Node], asm: &Assembly) -> InversionResult<Node> {
    let mut node = Node::empty();
    let mut curr = input;
    let mut error = Generic;
    'find_pattern: loop {
        for pattern in UN_PATTERNS {
            match pattern.extract(curr, asm) {
                Ok((new, inv)) => {
                    dbgln!("matched pattern {pattern:?}\n  on {curr:?}\n  to {inv:?}");
                    node.prepend(inv);
                    if new.is_empty() {
                        dbgln!("un-inverted\n  {input:?}\n  to {node:?}");
                        return Ok(node);
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

fn anti_inverse(input: &[Node], asm: &Assembly) -> InversionResult<Node> {
    let mut curr = input;
    let mut error = Generic;
    for pattern in ANTI_PATTERNS {
        match pattern.extract(curr, asm) {
            Ok((new, mut anti_inv)) => {
                dbgln!("matched pattern {pattern:?}\n  on {curr:?}\n  to {anti_inv:?}");
                curr = new;
                if curr.is_empty() {
                    dbgln!("anti-inverted\n  {input:?}\n  to {anti_inv:?}");
                    return Ok(anti_inv);
                }
                let rest_inv = un_inverse(curr, asm)?;
                anti_inv.push(rest_inv);
                dbgln!("anti-inverted\n  {input:?}\n  to {anti_inv:?}");
                return Ok(anti_inv);
            }
            Err(e) => error = error.max(e),
        }
    }
    Err(error)
}

static UN_PATTERNS: &[&dyn InvertPattern] = &[
    &PrimPat,
    &ImplPrimPat,
    &ArrayPat,
    &UnpackPat,
    &DipPat,
    &(Sqrt, (Dup, Mul)),
    &((Dup, Add), (2, Div)),
    &((Dup, Mul), Sqrt),
    &(Select, (Dup, Len, Range)),
    &(Pick, (Dup, Shape, Range)),
    &(Orient, (Dup, Shape, Len, Range)),
    &(Sort, UnSort),
    &(SortDown, UnSort),
    &((Val, ValidateType), ValidateType),
    &((Val, TagVariant), ValidateVariant),
    &((Val, ValidateVariant), TagVariant),
    &(
        (Val, Min),
        (
            Over,
            Lt,
            Deshape,
            Deduplicate,
            Not,
            0,
            Complex,
            crate::Complex::ONE,
            MatchPattern,
        ),
    ),
    &(
        (Val, Max),
        (
            Over,
            Gt,
            Deshape,
            Deduplicate,
            Not,
            0,
            Complex,
            crate::Complex::ONE,
            MatchPattern,
        ),
    ),
];

static ANTI_PATTERNS: &[&dyn InvertPattern] = &[
    &(Complex, (crate::Complex::I, Mul, Sub)),
    &(Atan, (Flip, UnAtan, Div, Mul)),
    &((IgnoreMany(Flip), Add), Sub),
    &(Sub, Add),
    &((Flip, Sub), (Flip, Sub)),
    &((IgnoreMany(Flip), Mul), Div),
    &(Div, Mul),
    &((Flip, Div), (Flip, Div)),
    &(Rotate, (Neg, Rotate)),
    &((Neg, Rotate), Rotate),
    &(Pow, (1, Flip, Div, Pow)),
    &((Flip, Pow), Log),
    &(Log, (Flip, Pow)),
    &((Flip, Log), (Flip, 1, Flip, Div, Pow)),
    &(Min, Min),
    &(Max, Max),
    &(Orient, UndoOrient),
    &(Drop, AntiDrop),
    &(Select, AntiSelect),
    &(Pick, AntiPick),
    &(Base, UndoBase),
];

static UN_BY_PATTERNS: &[&dyn InvertPattern] = &[
    &((IgnoreMany(Flip), Add), (Flip, Sub)),
    &(Sub, Sub),
    &((Flip, Sub), Add),
    &((IgnoreMany(Flip), Mul), (Flip, Div)),
    &(Div, Div),
    &((Flip, Div), Mul),
    &(Pow, (Flip, Log)),
    &(Log, (1, Flip, Div, Pow)),
    &((Flip, Log), Pow),
    &(Min, Min),
    &(Max, Max),
    &(Select, IndexOf),
    &(IndexOf, Select),
];

pub trait InvertPattern: fmt::Debug + Sync {
    fn extract<'a>(&self, input: &'a [Node], asm: &Assembly)
        -> InversionResult<(&'a [Node], Node)>;
}

macro_rules! inverse {
    ($name:ident, $input:ident, $asm:tt, $body:expr) => {
        #[derive(Debug)]
        struct $name;
        impl InvertPattern for $name {
            fn extract<'a>(
                &self,
                $input: &'a [Node],
                $asm: &Assembly,
            ) -> InversionResult<(&'a [Node], Node)> {
                $body
            }
        }
    };
    ($name:ident, $input:ident, $asm:tt, ref, $pat:pat, $body:expr) => {
        inverse!($name, $input, $asm, {
            let [$pat, ref $input @ ..] = $input else {
                return generic();
            };
            $body
        });
    };
    ($name:ident, $input:ident, $asm:tt, $pat:pat, $body:expr) => {
        inverse!($name, $input, $asm, {
            let &[$pat, ref $input @ ..] = $input else {
                return generic();
            };
            $body
        });
    };
    ($name:ident, $input:ident, $asm:tt, $prim:ident, $span:ident, $args:pat, $body:expr) => {
        inverse!($name, $input, $asm, ref, Mod($prim, args, $span), {
            let $args = args.as_slice() else {
                return generic();
            };
            let $span = *$span;
            $body
        });
    };
}

inverse!(
    ArrayPat,
    input,
    asm,
    ref,
    Array {
        len,
        inner,
        boxed,
        span
    },
    {
        let mut inv = un_inverse(inner.as_slice(), asm)?;
        inv.prepend(Node::Unpack {
            count: *len,
            unbox: *boxed,
            span: *span,
        });
        Ok((input, inv))
    }
);

inverse!(DipPat, input, asm, Dip, span, [f], {
    let inv = f.un_inverse(asm)?;
    Ok((input, Mod(Dip, eco_vec![inv], span)))
});

inverse!(OnPat, input, asm, Dip, span, [f], {
    let inv = f.anti_inverse(asm)?;
    Ok((input, Mod(On, eco_vec![inv], span)))
});

inverse!(
    UnpackPat,
    input,
    asm,
    Unpack {
        count,
        unbox,
        span,
        ..
    },
    {
        let mut inv = un_inverse(input, asm)?;
        inv.push(Array {
            len: count,
            inner: Node::empty().into(),
            boxed: unbox,
            span,
        });
        Ok((&[], inv))
    }
);

inverse!(PrimPat, input, _, Prim(prim, span), {
    let inv = match prim {
        // Basic
        Identity => Prim(Identity, span),
        Flip => Prim(Flip, span),
        Pop => ImplPrim(UnPop, span),
        Neg => Prim(Neg, span),
        Not => Prim(Not, span),
        Sin => ImplPrim(Asin, span),
        Atan => ImplPrim(UnAtan, span),
        Complex => ImplPrim(UnComplex, span),
        Reverse => Prim(Reverse, span),
        Transpose => ImplPrim(TransposeN(-1), span),
        Bits => ImplPrim(UnBits, span),
        Couple => ImplPrim(UnCouple, span),
        Box => ImplPrim(UnBox, span),
        Where => ImplPrim(UnWhere, span),
        Utf8 => ImplPrim(UnUtf, span),
        Graphemes => ImplPrim(UnGraphemes, span),
        Parse => ImplPrim(UnParse, span),
        Fix => ImplPrim(UnFix, span),
        Shape => ImplPrim(UnShape, span),
        Map => ImplPrim(UnMap, span),
        Stack => ImplPrim(UnStack, span),
        Keep => ImplPrim(UnKeep, span),
        GifEncode => ImplPrim(GifDecode, span),
        AudioEncode => ImplPrim(AudioDecode, span),
        ImageEncode => ImplPrim(ImageDecode, span),
        Sys(SysOp::Clip) => ImplPrim(UnClip, span),
        Sys(SysOp::RawMode) => ImplPrim(UnRawMode, span),
        Json => ImplPrim(UnJson, span),
        Csv => ImplPrim(UnCsv, span),
        Xlsx => ImplPrim(UnXlsx, span),
        Fft => ImplPrim(UnFft, span),
        DateTime => ImplPrim(UnDatetime, span),
        Trace => ImplPrim(
            TraceN {
                n: 1,
                inverse: true,
                stack_sub: false,
            },
            span,
        ),
        _ => return generic(),
    };
    Ok((input, inv))
});

inverse!(ImplPrimPat, input, _, ImplPrim(prim, span), {
    let inv = match prim {
        UnPop => Prim(Pop, span),
        Asin => Prim(Sin, span),
        TransposeN(n) => ImplPrim(TransposeN(-n), span),
        UnWhere => Prim(Where, span),
        UnUtf => Prim(Utf8, span),
        UnGraphemes => Prim(Graphemes, span),
        UnAtan => Prim(Atan, span),
        UnComplex => Prim(Complex, span),
        UnCouple => Prim(Couple, span),
        UnParse => Prim(Parse, span),
        UnFix => Prim(Fix, span),
        UnShape => Prim(Shape, span),
        UnMap => Prim(Map, span),
        UnStack => Prim(Stack, span),
        UnJoin => Prim(Join, span),
        UnKeep => Prim(Keep, span),
        UnBox => Prim(Box, span),
        UnJson => Prim(Json, span),
        UnCsv => Prim(Csv, span),
        UnXlsx => Prim(Xlsx, span),
        UnFft => Prim(Fft, span),
        ImageDecode => Prim(ImageEncode, span),
        GifDecode => Prim(GifEncode, span),
        AudioDecode => Prim(AudioEncode, span),
        UnDatetime => Prim(DateTime, span),
        UnRawMode => Prim(Sys(SysOp::RawMode), span),
        UnClip => Prim(Sys(SysOp::Clip), span),
        TraceN {
            n,
            inverse,
            stack_sub,
        } => ImplPrim(
            TraceN {
                n,
                inverse: !inverse,
                stack_sub,
            },
            span,
        ),
        _ => return generic(),
    };
    Ok((input, inv))
});

inverse!(Val, input, asm, {
    for end in (1..=input.len()).rev() {
        let chunk = &input[..end];
        if let Some(sig) = nodes_clean_sig(chunk) {
            if sig == (0, 1) && chunk.iter().all(|n| n.is_pure(Purity::Pure, asm)) {
                return Ok((&input[end..], Node::from(chunk)));
            }
        }
    }
    generic()
});

impl SpanFromNodes for Val {
    fn span_from_nodes<'a>(
        &self,
        nodes: &'a [Node],
        asm: &Assembly,
    ) -> Option<(&'a [Node], Option<usize>)> {
        Some((self.extract(nodes, asm).ok()?.0, None))
    }
}

impl<A, B> InvertPattern for (A, B)
where
    A: SpanFromNodes,
    B: AsNode,
{
    fn extract<'a>(
        &self,
        input: &'a [Node],
        asm: &Assembly,
    ) -> InversionResult<(&'a [Node], Node)> {
        let (a, b) = self;
        let (input, span) = a.span_from_nodes(input, asm).ok_or(Generic)?;
        let span = span.ok_or(Generic)?;
        Ok((input, b.as_node(span)))
    }
}

#[derive(Debug)]
struct IgnoreMany<T>(T);
impl<T> SpanFromNodes for IgnoreMany<T>
where
    T: SpanFromNodes,
{
    fn span_from_nodes<'a>(
        &self,
        mut nodes: &'a [Node],
        asm: &Assembly,
    ) -> Option<(&'a [Node], Option<usize>)> {
        let mut span = None;
        while let Some((nds, sp)) = self.0.span_from_nodes(nodes, asm) {
            nodes = nds;
            span = span.or(sp);
        }
        Some((nodes, span))
    }
}
