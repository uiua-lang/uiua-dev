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
    /// Get the contra inverse of this node
    fn contra_inverse(&self, asm: &Assembly) -> InversionResult<Node> {
        dbgln!("contra-inverting {self:?}");
        contra_inverse(self.as_slice(), asm)
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
    /// Get the contra inverse of this node
    fn contra_inverse(&self, asm: &Assembly) -> InversionResult<SigNode> {
        let inv = self.node.contra_inverse(asm)?;
        let sig = self.sig.anti().ok_or(Generic)?;
        Ok(SigNode::new(inv, sig))
    }
}

pub fn un_inverse(input: &[Node], asm: &Assembly) -> InversionResult<Node> {
    let mut node = Node::empty();
    let mut curr = input;
    let mut error = Generic;
    'find_pattern: loop {
        for pattern in UN_PATTERNS {
            match pattern.invert_extract(curr, asm) {
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

pub fn anti_inverse(input: &[Node], asm: &Assembly) -> InversionResult<Node> {
    let mut curr = input;
    let mut error = Generic;
    for pattern in ANTI_PATTERNS {
        match pattern.invert_extract(curr, asm) {
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

pub fn contra_inverse(input: &[Node], asm: &Assembly) -> InversionResult<Node> {
    let mut curr = input;
    let mut error = Generic;
    for pattern in CONTRA_PATTERNS {
        match pattern.invert_extract(curr, asm) {
            Ok((new, mut contra_inv)) => {
                dbgln!("matched pattern {pattern:?}\n  on {curr:?}\n  to {contra_inv:?}");
                curr = new;
                if curr.is_empty() {
                    dbgln!("contra-inverted\n  {input:?}\n  to {contra_inv:?}");
                    return Ok(contra_inv);
                }
                let rest_inv = un_inverse(curr, asm)?;
                contra_inv.push(rest_inv);
                dbgln!("contra-inverted\n  {input:?}\n  to {contra_inv:?}");
                return Ok(contra_inv);
            }
            Err(e) => error = error.max(e),
        }
    }
    Err(error)
}

pub static UN_PATTERNS: &[&dyn InvertPattern] = &[
    &InnerAnti,
    &InnerAntiDip,
    &PrimPat,
    &ImplPrimPat,
    &ArrayPat,
    &UnpackPat,
    &DipPat,
    &OnPat,
    &ByPat,
    &RowsPat,
    &Trivial,
    &ScanPat,
    &ReduceMulPat,
    &PrimesPat,
    &JoinPat,
    &CustomPat,
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
    &(Dup, (Over, Flip, MatchPattern)),
    &MatchConst,
];

pub static ANTI_PATTERNS: &[&dyn InvertPattern] = &[
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
    &(Complex, (crate::Complex::I, Mul, Sub)),
    &(Min, MatchGe),
    &(Max, MatchLe),
    &(Orient, AntiOrient),
    &(Drop, AntiDrop),
    &(Select, AntiSelect),
    &(Pick, AntiPick),
    &(Base, AntiBase),
    &AntiRepeatPat,
    &AntiInsertPat,
    &AntiJoinPat,
    &AntiCustomPat,
];

pub static CONTRA_PATTERNS: &[&dyn InvertPattern] = &[
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
    fn invert_extract<'a>(
        &self,
        input: &'a [Node],
        asm: &Assembly,
    ) -> InversionResult<(&'a [Node], Node)>;
}

macro_rules! inverse {
    // Optional parens
    ($(#[$attr:meta])* $($doc:literal,)? ($($tt:tt)*), $body:expr) => {
        inverse!($(#[$attr])* $($doc,)? $($tt)*, $body);
    };
    ($(#[$attr:meta])* $($doc:literal,)? ($($tt:tt)*), ref, $pat:pat, $body:expr) => {
        inverse!($(#[$attr])* $($doc,)? $($tt)*, ref, $pat, $body);
    };
    ($(#[$attr:meta])* $($doc:literal,)? ($($tt:tt)*), $pat:pat, $body:expr) => {
        inverse!($(#[$attr])* $($doc,)? $($tt)*, $pat, $body);
    };
    // Main impl
    ($(#[$attr:meta])* $($doc:literal,)? $name:ident, $input:ident, $asm:tt, $body:expr) => {
        #[derive(Debug)]
        pub(crate) struct $name;
        impl InvertPattern for $name {
            fn invert_extract<'a>(
                &self,
                $input: &'a [Node],
                $asm: &Assembly,
            ) -> InversionResult<(&'a [Node], Node)> {
                $body
            }
        }
    };
    ($(#[$attr:meta])* $($doc:literal,)? $name:ident, $input:ident, $asm:tt, ref, $pat:pat, $body:expr) => {
        inverse!($([$attr])* $($doc)? $name, $input, $asm, {
            let [$pat, ref $input @ ..] = $input else {
                return generic();
            };
            $body
        });
    };
    ($(#[$attr:meta])* $($doc:literal,)? $name:ident, $input:ident, $asm:tt, $pat:pat, $body:expr) => {
        inverse!($([$attr])* $($doc)? $name, $input, $asm, {
            let &[$pat, ref $input @ ..] = $input else {
                return generic();
            };
            $body
        });
    };
    ($(#[$attr:meta])* $($doc:literal,)? $name:ident, $input:ident, $asm:tt, $prim:ident, $span:ident, $args:pat, $body:expr) => {
        inverse!($([$attr])* $($doc)? $name, $input, $asm, ref, Mod($prim, args, $span), {
            let $args = args.as_slice() else {
                return generic();
            };
            let $span = *$span;
            $body
        });
    };
}

inverse!(
    (ArrayPat, input, asm),
    ref,
    Array {
        len,
        inner,
        boxed,
        span
    },
    {
        match len {
            ArrayLen::Static(len) => {
                let mut inv = un_inverse(inner.as_slice(), asm)?;
                inv.prepend(Node::Unpack {
                    count: *len,
                    unbox: *boxed,
                    span: *span,
                });
                Ok((input, inv))
            }
            ArrayLen::Dynamic(_) => generic(),
        }
    }
);

inverse!(DipPat, input, asm, Dip, span, [f], {
    let inv = f.un_inverse(asm)?;
    Ok((input, Mod(Dip, eco_vec![inv], span)))
});

inverse!(OnPat, input, asm, On, span, [f], {
    let inv = f.anti_inverse(asm)?;
    Ok((input, Mod(On, eco_vec![inv], span)))
});

inverse!(ByPat, input, asm, Dip, span, [f], {
    let inv = f.contra_inverse(asm)?;
    Ok((input, Mod(By, eco_vec![inv], span)))
});

inverse!("Match a constant exactly", (MatchConst, input, asm), {
    let (input, mut val) = Val.invert_extract(input, asm)?;
    val.push(ImplPrim(MatchPattern, asm.spans.len() - 1));
    Ok((input, val))
});

inverse!(
    "Matches an anti inverse with the first argument as part of the input",
    (InnerAnti, input, asm),
    {
        let (input, mut node) = Val.invert_extract(input, asm)?;
        for end in 1..=input.len() {
            if let Ok(inv) = anti_inverse(&input[..end], asm) {
                node.push(inv);
                return Ok((&input[end..], node));
            }
        }
        generic()
    }
);

inverse!((InnerAntiDip, input, asm), {
    let [Mod(Dip, args, dip_span), input @ ..] = input else {
        return generic();
    };
    let args: Vec<_> = args.iter().map(|sn| sn.node.clone()).collect();
    let ([], mut node) = Val.invert_extract(&args, asm)? else {
        return generic();
    };
    node.push(Prim(Flip, *dip_span));
    node.extend(input.iter().cloned());
    let inv = node.un_inverse(asm)?;
    Ok((&[], inv))
});

inverse!(
    (UnpackPat, input, asm),
    Unpack {
        count,
        unbox,
        span,
        ..
    },
    {
        let mut inv = un_inverse(input, asm)?;
        inv.push(Array {
            len: ArrayLen::Static(count),
            inner: Node::empty().into(),
            boxed: unbox,
            span,
        });
        Ok((&[], inv))
    }
);

inverse!(RowsPat, input, asm, Rows, span, [f], {
    Ok((input, Mod(Rows, eco_vec![f.un_inverse(asm)?], span)))
});

inverse!(AntiInsertPat, input, asm, {
    let (input, val) = Val.invert_extract(input, asm)?;
    let &[Prim(Insert, span), ref input @ ..] = input else {
        return generic();
    };
    let inv = Node::from_iter([
        PushUnder(1, span),
        val,
        Prim(Over, span),
        Prim(Over, span),
        Prim(Has, span),
        Node::new_push(1),
        ImplPrim(MatchPattern, span),
        Prim(Over, span),
        Prim(Over, span),
        Prim(Get, span),
        PushUnder(1, span),
        Prim(Remove, span),
        PopUnder(2, span),
        ImplPrim(MatchPattern, span),
    ]);
    Ok((input, inv))
});

inverse!(ScanPat, input, asm, {
    let un = matches!(input, [ImplMod(UnScan, ..), ..]);
    let ([Mod(Scan, args, span), input @ ..] | [ImplMod(UnScan, args, span), input @ ..]) = input
    else {
        return generic();
    };
    let [f] = args.as_slice() else {
        return generic();
    };
    let inverse = match f.node.as_primitive() {
        Some(Primitive::Add) if !un => Prim(Sub, *span),
        Some(Primitive::Mul) if !un => Prim(Div, *span),
        Some(Primitive::Sub) if un => Prim(Add, *span),
        Some(Primitive::Div) if un => Prim(Mul, *span),
        Some(Primitive::Eq) => Prim(Eq, *span),
        Some(Primitive::Ne) => Prim(Ne, *span),
        _ => f.node.un_inverse(asm)?,
    }
    .sig_node()?;
    let inverse = if un {
        Mod(Scan, eco_vec![inverse], *span)
    } else {
        ImplMod(UnScan, eco_vec![inverse], *span)
    };
    Ok((input, inverse))
});

inverse!(
    (ReduceMulPat, input, _, Reduce, span),
    [SigNode {
        node: Prim(Mul, _),
        ..
    }],
    Ok((input, ImplPrim(Primes, span)))
);

inverse!(
    (PrimesPat, input, _, ImplPrim(Primes, span)),
    Ok((
        input,
        Mod(Reduce, eco_vec![Prim(Mul, span).sig_node()?], span)
    ))
);

inverse!(
    (AntiRepeatPat, input, _),
    ref,
    ImplMod(RepeatWithInverse, args, span),
    {
        let [f, inv] = args.as_slice() else {
            return generic();
        };
        Ok((
            input,
            ImplMod(RepeatWithInverse, eco_vec![inv.clone(), f.clone()], *span),
        ))
    }
);

inverse!(JoinPat, input, _, {
    Ok(match input {
        [Prim(Join, span), input @ ..] => (input, ImplPrim(UnJoin, *span)),
        [Prim(Flip, span), Prim(Join, _), input @ ..] => (input, ImplPrim(UnJoinEnd, *span)),
        [Prim(Couple, span), Prim(Flip, _), Prim(Join, _), input @ ..] => {
            let inv = Node::from_iter([Node::new_push([2]), ImplPrim(UnJoinShape, *span)]);
            (input, inv)
        }
        [Prim(Couple, span), Prim(Join, _), input @ ..] => {
            let inv = Node::from_iter([Node::new_push([2]), ImplPrim(UnJoinShapeEnd, *span)]);
            (input, inv)
        }
        _ => return generic(),
    })
});

inverse!(AntiJoinPat, input, _, {
    Ok(match *input {
        [Prim(Join, span), ref input @ ..] => {
            let inv = Node::from_iter([
                Prim(Dup, span),
                Prim(Shape, span),
                Prim(Flip, span),
                PushUnder(1, span),
                ImplPrim(UnJoinShape, span),
                PopUnder(1, span),
                ImplPrim(MatchPattern, span),
            ]);
            (input, inv)
        }
        [Prim(Flip, span), Prim(Join, _), ref input @ ..] => {
            let inv = Node::from_iter([
                Prim(Dup, span),
                Prim(Shape, span),
                Prim(Flip, span),
                PushUnder(1, span),
                ImplPrim(UnJoinShapeEnd, span),
                PopUnder(1, span),
                ImplPrim(MatchPattern, span),
            ]);
            (input, inv)
        }
        _ => return generic(),
    })
});

inverse!(CustomPat, input, _, ref, CustomInverse(cust, span), {
    let mut cust = cust.clone();
    let un = cust.un.take().ok_or(Generic)?;
    cust.un = Some(cust.normal);
    cust.normal = un;
    cust.anti = None;
    cust.under = None;
    Ok((input, CustomInverse(cust, *span)))
});

inverse!(AntiCustomPat, input, _, ref, CustomInverse(cust, span), {
    let mut cust = cust.clone();
    let anti = cust.anti.take().ok_or(Generic)?;
    cust.anti = Some(cust.normal);
    cust.normal = anti;
    cust.un = None;
    cust.under = None;
    Ok((input, CustomInverse(cust, *span)))
});

#[derive(Debug)]
pub(crate) struct Trivial;
impl InvertPattern for Trivial {
    fn invert_extract<'a>(
        &self,
        input: &'a [Node],
        asm: &Assembly,
    ) -> InversionResult<(&'a [Node], Node)> {
        match input {
            [NoInline(inner), input @ ..] => Ok((input, NoInline(inner.un_inverse(asm)?.into()))),
            [TrackCaller(inner), input @ ..] => {
                Ok((input, TrackCaller(inner.un_inverse(asm)?.into())))
            }
            [node @ SetOutputComment { .. }, input @ ..] => Ok((input, node.clone())),
            [Call(f, _), input @ ..] => Ok((input, asm[f].un_inverse(asm).map_err(|e| e.func(f))?)),
            _ => generic(),
        }
    }
}

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
        Some((self.invert_extract(nodes, asm).ok()?.0, None))
    }
}

impl<A, B> InvertPattern for (A, B)
where
    A: SpanFromNodes,
    B: AsNode,
{
    fn invert_extract<'a>(
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
