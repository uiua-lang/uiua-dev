use crate::check::nodes_sig;

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
    if input.is_empty() {
        return Ok(Node::empty());
    }
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
    // An anti inverse can be optionaly sandwhiched by an un inverse on either side
    let mut curr = input;
    let mut error = Generic;

    // Anti inverse
    let mut got_anti = false;
    let mut anti = Node::empty();
    let mut start = 0;
    'find_anti: for s in (0..input.len()).rev() {
        error = Generic;
        curr = &input[s..];
        for pattern in ANTI_PATTERNS {
            match pattern.invert_extract(curr, asm) {
                Ok((new, anti_inv)) => {
                    dbgln!("matched pattern {pattern:?}\n  on {curr:?}\n  to {anti_inv:?}");
                    curr = new;
                    anti.prepend(anti_inv);
                    got_anti = true;
                    start = s;
                    break 'find_anti;
                }
                Err(e) => error = error.max(e),
            }
        }
    }
    if !got_anti {
        return Err(error);
    }

    // Leading non-inverted section
    let pre = Node::from(&input[..start]);
    let pre_sig = pre.sig()?;
    if !(pre_sig == (0, 1) || pre_sig.args == pre_sig.outputs) {
        return generic();
    }
    if !pre.is_empty() {
        dbgln!("leading non-inverted section:\n  {pre:?}");
    }

    // Trailing un inverse
    let mut post = Node::empty();
    'outer: loop {
        for pattern in UN_PATTERNS {
            match pattern.invert_extract(curr, asm) {
                Ok((new, un_inv)) => {
                    dbgln!("matched pattern {pattern:?}\n  on {curr:?}\n  to {un_inv:?}");
                    curr = new;
                    post.prepend(un_inv);
                    continue 'outer;
                }
                Err(e) => error = error.max(e),
            }
        }
        break;
    }
    if !post.is_empty() {
        let span = post
            .span()
            .or_else(|| anti.span())
            .or_else(|| pre.span())
            .ok_or(Generic)?;
        post = Mod(Dip, eco_vec![post.sig_node()?], span);
    }

    anti.prepend(post);
    anti.prepend(pre);

    dbgln!("anti-inverted\n     {input:?}\n  to {anti:?}");
    Ok(anti)
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
    &InnerContraDip,
    &JoinPat,
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
    &CustomPat,
    &FormatPat,
    &FillPat,
    &InsertPat,
    &(Sqrt, (Dup, Mul)),
    &((Dup, Add), (2, Div)),
    &((Dup, Mul), Sqrt),
    &(Select, (Dup, Len, Range)),
    &(Pick, (Dup, Shape, Range)),
    &(Orient, (Dup, Shape, Len, Range)),
    &(Sort, UnSort),
    &(SortDown, UnSort),
    &RequireVal((ValidateType, ValidateType)),
    &RequireVal((TagVariant, ValidateVariant)),
    &RequireVal((ValidateVariant, TagVariant)),
    &(Dup, (Over, Flip, MatchPattern)),
    &PrimPat,
    &ImplPrimPat,
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
    &(Min, MatchLe),
    &(Max, MatchGe),
    &(Orient, AntiOrient),
    &(Drop, AntiDrop),
    &(Select, AntiSelect),
    &(Pick, AntiPick),
    &(Base, AntiBase),
    &AntiTrivial,
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
    &((Flip, Pow), (Flip, 1, Flip, Div, Pow)),
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
            #[allow(irrefutable_let_patterns)]
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
    let mut inv = f.anti_inverse(asm)?;
    if f.sig.args == f.sig.outputs {
        inv = Mod(Dip, eco_vec![inv], span).sig_node()?;
    }
    let mut inv = Mod(On, eco_vec![inv], span);
    if f.sig.args == f.sig.outputs {
        inv.push(ImplPrim(MatchPattern, span))
    }
    Ok((input, inv))
});

inverse!(ByPat, input, asm, By, span, [f], {
    // Under's undo step
    if f.sig.args == 1 {
        if let Ok((before, after)) = f.node.under_inverse(Signature::new(1, 1), asm) {
            let mut inv = before;
            (0..f.sig.outputs).for_each(|_| inv.push(Prim(Pop, span)));
            for _ in 0..f.sig.outputs {
                inv = Mod(Dip, eco_vec![inv.sig_node()?], span);
            }
            inv.push(after);
            return Ok((input, inv));
        }
    }
    // Contra inverse
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

inverse!((InnerContraDip, input, asm), {
    let [Mod(Dip, args, dip_span), input @ ..] = input else {
        return generic();
    };
    let args: Vec<_> = args.iter().map(|sn| sn.node.clone()).collect();
    let ([], val) = Val.invert_extract(&args, asm)? else {
        return generic();
    };
    let mut contra = contra_inverse(input, asm)?;
    contra.prepend(Prim(Flip, *dip_span));
    contra.prepend(val);
    Ok((&[], contra))
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

inverse!(JoinPat, input, asm, {
    let orig_input = input;
    let mut input = input;
    let Some((join_index, join_span)) = (input.iter().enumerate().rev())
        .filter_map(|(i, node)| match node {
            Prim(Join, span) => Some((i, *span)),
            _ => None,
        })
        .find(|(i, _)| nodes_clean_sig(&input[..*i]).is_some())
    else {
        return generic();
    };
    let mut dipped = false;
    let node = if let Some((inp, mut node)) = Val.invert_extract(input, asm).ok().or_else(|| {
        let [Mod(Dip, args, _), input @ ..] = input else {
            return None;
        };
        let [inner] = args.as_slice() else {
            return None;
        };
        let Ok(([], val)) = Val.invert_extract(inner.node.as_slice(), asm) else {
            return None;
        };
        dipped = true;
        Some((input, val))
    }) {
        input = inp;
        if let Some(i) = (1..=input.len())
            .rev()
            .find(|&i| nodes_clean_sig(&input[..i]).is_some_and(|sig| sig == (0, 0)))
        {
            node.extend(un_inverse(&input[..i], asm)?);
            input = &input[i..];
        }
        let (prim, span) = match *input {
            [Prim(Join, span), ref inp @ ..] if dipped => {
                input = inp;
                (UnJoinShapeEnd, span)
            }
            [Prim(Join, span), ref inp @ ..] => {
                input = inp;
                (UnJoinShape, span)
            }
            [Prim(Flip, _), Prim(Join, span), ref inp @ ..] if !dipped => {
                input = inp;
                (UnJoinShapeEnd, span)
            }
            _ => return generic(),
        };
        let inner =
            Node::from_iter([Prim(Primitive::Shape, span), ImplPrim(prim, span)]).sig_node()?;
        node.extend([
            Mod(Dip, eco_vec![inner], span),
            ImplPrim(MatchPattern, span),
        ]);
        node
    } else if let Some(i) = (0..join_index)
        .find(|&i| nodes_clean_sig(&input[i..join_index]).is_some_and(|sig| sig == (0, 1)))
    {
        let mut node = ImplPrim(UnJoin, join_span);
        node.extend(un_inverse(&input[i..join_index], asm)?);
        node.extend(un_inverse(&input[..i], asm)?);
        input = &input[join_index + 1..];
        node
    } else {
        fn invert_inner(mut input: &[Node], asm: &Assembly) -> InversionResult<Node> {
            let mut node = Node::empty();
            while !input.is_empty() {
                if let [Mod(Dip, args, _), inp @ ..] = input {
                    let [inner] = args.as_slice() else {
                        return generic();
                    };
                    node.extend(invert_inner(inner.node.as_slice(), asm)?);
                    input = inp;
                    continue;
                }
                if let Some((i, _)) = input.iter().enumerate().skip(1).find(|(i, node)| {
                    nodes_clean_sig(&input[..*i]).is_some() && matches!(node, Mod(Dip, ..))
                }) {
                    node.extend(un_inverse(&input[..i], asm)?);
                    input = &input[i..];
                    continue;
                }
                node.extend(un_inverse(input, asm)?);
                break;
            }
            Ok(node)
        }
        let flip_after = join_index > 0 && matches!(input[join_index - 1], Prim(Flip, _));
        let flip_before = join_index > 1 && matches!(input[0], Prim(Flip, _));
        let flip = flip_before ^ flip_after;
        let before = &input[flip_before as usize..join_index - flip_after as usize];
        input = &input[join_index + 1..];
        let before_inv = invert_inner(before, asm)?;
        let before_sig = nodes_clean_sig(&before_inv).ok_or(Generic)?;
        let mut node = Node::empty();
        let count = before_sig.outputs.saturating_sub(before_sig.args) + 1;
        let prim = if count <= 1 {
            if flip {
                UnJoinEnd
            } else {
                UnJoin
            }
        } else {
            node.push(Push(count.into()));
            if flip {
                UnJoinShapeEnd
            } else {
                UnJoinShape
            }
        };
        node.push(ImplPrim(prim, join_span));
        node.push(before_inv);
        node
    };
    let orig_sig = nodes_sig(&orig_input[..orig_input.len() - input.len()])?;
    let inverted_sig = node.sig()?;
    if orig_sig.inverse() != inverted_sig {
        return generic();
    }
    Ok((input, node))
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
    cust.un = cust.normal.ok();
    cust.normal = Ok(un);
    cust.anti = None;
    cust.under = None;
    Ok((input, CustomInverse(cust, *span)))
});

inverse!(AntiCustomPat, input, _, ref, CustomInverse(cust, span), {
    let mut cust = cust.clone();
    let anti = cust.anti.take().ok_or(Generic)?;
    cust.anti = cust.normal.ok();
    cust.normal = Ok(anti);
    cust.un = None;
    cust.under = None;
    Ok((input, CustomInverse(cust, *span)))
});

inverse!(FormatPat, input, _, ref, Format(parts, span), {
    Ok((input, MatchFormatPattern(parts.clone(), *span)))
});

inverse!(FillPat, input, asm, Fill, span, [fill, f], {
    if fill.sig != (0, 1) {
        return generic();
    }
    let inv = f.un_inverse(asm)?;
    Ok((input, ImplMod(UnFill, eco_vec![fill.clone(), inv], span)))
});

inverse!(InsertPat, input, asm, {
    let (input, first) = Val.invert_extract(input, asm)?;
    let second = Val.invert_extract(input, asm);
    let &[Prim(Insert, span), ref input @ ..] = input else {
        return generic();
    };
    let (input, key, value) = if let Ok((input, key)) = second {
        (input, key, Some(first))
    } else {
        (input, first, None)
    };
    let mut node = Node::from_iter([
        key,
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
        PopUnder(1, span),
    ]);
    if let Some(value) = value {
        node.extend(value);
        node.push(ImplPrim(MatchPattern, span));
    }
    Ok((input, node))
});

inverse!(AntiInsertPat, input, _, Prim(Insert, span), {
    let args = eco_vec![Prim(Get, span).sig_node()?, Prim(Remove, span).sig_node()?];
    let inv = Mod(Fork, args, span);
    Ok((input, inv))
});

#[derive(Debug)]
struct Trivial;
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
            [Label(label, span), input @ ..] => {
                Ok((input, RemoveLabel(Some(label.clone()), *span)))
            }
            [RemoveLabel(Some(label), span), input @ ..] => {
                Ok((input, Label(label.clone(), *span)))
            }
            [node @ SetOutputComment { .. }, input @ ..] => Ok((input, node.clone())),
            [Call(f, _), input @ ..] => Ok((input, asm[f].un_inverse(asm).map_err(|e| e.func(f))?)),
            _ => generic(),
        }
    }
}

#[derive(Debug)]
struct AntiTrivial;
impl InvertPattern for AntiTrivial {
    fn invert_extract<'a>(
        &self,
        input: &'a [Node],
        asm: &Assembly,
    ) -> InversionResult<(&'a [Node], Node)> {
        match input {
            [NoInline(inner), input @ ..] => Ok((input, NoInline(inner.anti_inverse(asm)?.into()))),
            [TrackCaller(inner), input @ ..] => {
                Ok((input, TrackCaller(inner.anti_inverse(asm)?.into())))
            }
            [node @ SetOutputComment { .. }, input @ ..] => Ok((input, node.clone())),
            [Call(f, _), input @ ..] => {
                Ok((input, asm[f].anti_inverse(asm).map_err(|e| e.func(f))?))
            }
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

impl<P: InvertPattern> InvertPattern for MaybeVal<P> {
    fn invert_extract<'a>(
        &self,
        input: &'a [Node],
        asm: &Assembly,
    ) -> InversionResult<(&'a [Node], Node)> {
        let (input, val) = if let Ok((input, val)) = Val.invert_extract(input, asm) {
            (input, Some(val))
        } else {
            (input, None)
        };
        let (input, mut inv) = self.0.invert_extract(input, asm)?;
        if let Some(val) = val {
            inv.prepend(val);
        }
        Ok((input, inv))
    }
}

impl<P: InvertPattern> InvertPattern for RequireVal<P> {
    fn invert_extract<'a>(
        &self,
        input: &'a [Node],
        asm: &Assembly,
    ) -> InversionResult<(&'a [Node], Node)> {
        let (input, val) = Val.invert_extract(input, asm)?;
        let (input, mut inv) = self.0.invert_extract(input, asm)?;
        inv.prepend(val);
        Ok((input, inv))
    }
}
