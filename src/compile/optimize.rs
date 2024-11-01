use super::*;

use crate::{ImplPrimitive::*, Node::*, Primitive::*};

impl Node {
    pub(super) fn optimize(&mut self) {
        match self {
            Run(nodes) => {
                while OPTIMIZATIONS.iter().any(|op| op.match_and_replace(nodes)) {}
                if nodes.len() == 1 {
                    *self = take(nodes).remove(0);
                }
            }
            Mod(_, args, _) | ImplMod(_, args, _) => {
                for arg in args.make_mut() {
                    arg.node.optimize();
                }
            }
            CustomInverse(cust, _) => {
                if let Ok(normal) = cust.normal.as_mut() {
                    normal.node.optimize();
                }
                if let Some(un) = cust.un.as_mut() {
                    un.node.optimize();
                }
                if let Some(anti) = cust.anti.as_mut() {
                    anti.node.optimize();
                }
                if let Some((before, after)) = cust.under.as_mut() {
                    before.node.optimize();
                    after.node.optimize();
                }
            }
            _ => {}
        }
    }
}

static OPTIMIZATIONS: &[&dyn Optimization] = &[
    &((Reverse, First), Last),
    &((Rise, First), FirstMinIndex),
    &((Fall, Last), LastMinIndex),
    &((Fall, First), FirstMaxIndex),
    &((Rise, Last), LastMaxIndex),
    &((Where, First), FirstWhere),
    &((Where, Last), LastWhere),
    &((Where, Len), LenWhere),
    &((Range, MemberOf), MemberOfRange),
    &((Range, 1, Rerank, MemberOf), MultidimMemberOfRange),
    &((UnSort, Or(First, Last)), RandomRow),
    &((Deduplicate, Len), CountUnique),
    &((Or((Dup, Rise), M(By, Rise)), Select), Sort),
    &((Or((Dup, Fall), M(By, Fall)), Select), SortDown),
    &((Sort, Reverse), SortDown),
    &((SortDown, Reverse), Sort),
    &((Pop, Rand), ReplaceRand),
    &((Pop, Pop, Rand), ReplaceRand2),
    &TransposeOpt,
    &ReduceTableOpt,
    &ReduceDepthOpt,
    &AstarOpt,
    &PopConst,
    &TraceOpt,
];

opt!(PopConst, [Push(_), Prim(Pop, _)], []);

opt!(
    TransposeOpt,
    (
        [Prim(Transpose, span), Prim(Transpose, _)],
        ImplPrim(TransposeN(2), *span)
    ),
    (
        [ImplPrim(TransposeN(n), span), Prim(Transpose, _)],
        ImplPrim(TransposeN(n + 1), *span)
    ),
    (
        [ImplPrim(TransposeN(a), span), ImplPrim(TransposeN(b), _)],
        ImplPrim(TransposeN(a + b), *span)
    ),
);

opt!(
    ReduceTableOpt,
    [Mod(Table, table_args, span), Mod(Reduce, reduce_args, _)],
    ImplMod(
        ReduceTable,
        (table_args.iter().cloned())
            .chain(reduce_args.iter().cloned())
            .collect(),
        *span
    )
);

struct ReduceDepthOpt;
impl Optimization for ReduceDepthOpt {
    fn match_and_replace(&self, nodes: &mut EcoVec<Node>) -> bool {
        for i in 0..nodes.len() {
            let [Mod(Rows, args, _), ..] = &nodes[i..] else {
                continue;
            };
            let [f] = args.as_slice() else {
                continue;
            };
            match f.node.as_slice() {
                [Mod(Reduce, reduce_args, span)] => {
                    let impl_mod = ImplMod(ReduceDepth(1), reduce_args.clone(), *span);
                    replace_nodes(nodes, i, 1, impl_mod);
                    return true;
                }
                [ImplMod(ReduceDepth(depth), reduce_args, span)] => {
                    let impl_mod = ImplMod(ReduceDepth(depth + 1), reduce_args.clone(), *span);
                    replace_nodes(nodes, i, 1, impl_mod);
                    return true;
                }
                _ => {}
            }
        }
        false
    }
}

opt!(
    AstarOpt,
    (
        [Mod(Astar, args, span), Prim(First, _)],
        ImplMod(AstarFirst, args.clone(), *span)
    ),
    (
        [Mod(Astar, args, span), Prim(Pop, pop_span)],
        [
            ImplMod(AstarFirst, args.clone(), *span),
            Prim(Pop, *pop_span)
        ]
    ),
);

opt!(
    TraceOpt,
    (
        [Prim(Trace, span), Prim(Trace, _)],
        ImplPrim(
            TraceN {
                n: 2,
                inverse: false,
                stack_sub: false
            },
            *span
        )
    ),
    (
        [
            ImplPrim(
                TraceN {
                    n,
                    inverse: false,
                    ..
                },
                span
            ),
            Prim(Trace, _)
        ],
        ImplPrim(
            TraceN {
                n: n + 1,
                inverse: false,
                stack_sub: false
            },
            *span
        )
    ),
);

trait Optimization: Sync {
    fn match_and_replace(&self, nodes: &mut EcoVec<Node>) -> bool;
}

impl<A, B> Optimization for (A, B)
where
    A: OptPattern,
    B: OptReplace,
{
    fn match_and_replace(&self, nodes: &mut EcoVec<Node>) -> bool {
        for i in 0..nodes.len() {
            if let Some((n, Some(span))) = self.0.match_nodes(&nodes[i..]) {
                replace_nodes(nodes, i, n, self.1.replacement_node(span));
                return true;
            }
        }
        false
    }
}

trait OptPattern: Sync {
    fn match_nodes(&self, nodes: &[Node]) -> Option<(usize, Option<usize>)>;
}
impl OptPattern for Primitive {
    fn match_nodes(&self, nodes: &[Node]) -> Option<(usize, Option<usize>)> {
        match nodes {
            [Node::Prim(p, span), ..] if self == p => Some((1, Some(*span))),
            _ => None,
        }
    }
}
impl OptPattern for ImplPrimitive {
    fn match_nodes(&self, nodes: &[Node]) -> Option<(usize, Option<usize>)> {
        match nodes {
            [Node::ImplPrim(p, span), ..] if self == p => Some((1, Some(*span))),
            _ => None,
        }
    }
}
impl OptPattern for i32 {
    fn match_nodes(&self, nodes: &[Node]) -> Option<(usize, Option<usize>)> {
        match nodes {
            [Node::Push(n), ..] if n == self => Some((1, None)),
            _ => None,
        }
    }
}
struct Or<A, B>(A, B);
impl<A, B> OptPattern for Or<A, B>
where
    A: OptPattern,
    B: OptPattern,
{
    fn match_nodes(&self, nodes: &[Node]) -> Option<(usize, Option<usize>)> {
        self.0
            .match_nodes(nodes)
            .or_else(|| self.1.match_nodes(nodes))
    }
}

struct M<A, B>(A, B);
impl OptPattern for M<Primitive, Primitive> {
    fn match_nodes(&self, nodes: &[Node]) -> Option<(usize, Option<usize>)> {
        let [Mod(a, args, span), ..] = nodes else {
            return None;
        };
        let [f] = args.as_slice() else {
            return None;
        };
        if *a == self.0 && f.node.as_primitive() == Some(self.1) {
            Some((1, Some(*span)))
        } else {
            None
        }
    }
}

trait OptReplace: Sync {
    fn replacement_node(&self, span: usize) -> Node;
}
impl OptReplace for Primitive {
    fn replacement_node(&self, span: usize) -> Node {
        Node::Prim(*self, span)
    }
}
impl OptReplace for ImplPrimitive {
    fn replacement_node(&self, span: usize) -> Node {
        Node::ImplPrim(*self, span)
    }
}

fn replace_nodes(nodes: &mut EcoVec<Node>, i: usize, n: usize, new: Node) {
    // dbg!(&nodes, i, n, &new);
    let orig_len = nodes.len();
    debug_assert!(orig_len - n >= i);
    nodes.make_mut()[i..].rotate_left(n);
    nodes.truncate(orig_len - n);
    nodes.extend(new);
    let added = nodes.len() - (orig_len - n);
    nodes.make_mut()[i..].rotate_right(added);
}

macro_rules! opt {
    ($name:ident, [$($pat:pat),*], $new:expr) => {
        opt!($name, ([$($pat),*], $new));
    };
    ($name:ident, $(([$($pat:pat),*], $new:expr)),* $(,)?) => {
        struct $name;
        impl Optimization for $name {
            fn match_and_replace(&self, nodes: &mut EcoVec<Node>) -> bool {
                for i in 0..nodes.len() {
                    match &nodes[i..] {
                        $(
                            [$($pat),*, ..] => {
                                const N: usize = 0 $(+ {stringify!($pat); 1})*;
                                let new = $new;
                                replace_nodes(nodes, i, N, new.into());
                                return true;
                            }
                        )*
                        _ => {}
                    }
                }
                false
            }
        }
    };
}
use opt;

macro_rules! opt_pattern {
    ($($T:ident),*) => {
        impl<$($T),*> OptPattern for ($($T,)*)
        where
            $($T: OptPattern,)*
        {
            #[allow(unused, non_snake_case)]
            fn match_nodes(&self, mut nodes: &[Node]) -> Option<(usize, Option<usize>)> {
                let ($($T,)*) = self;
                let mut i = 0;
                let mut span = None;
                $(
                    let (n, sp) = $T.match_nodes(nodes)?;
                    i += n;
                    nodes = &nodes[n..];
                    if sp.is_some() {
                        span = sp;
                    }
                )*
                Some((i, span))
            }
        }
        #[allow(non_snake_case)]
        impl<$($T),*> OptReplace for ($($T,)*)
        where
            $($T: OptReplace,)*
        {
            fn replacement_node(&self, span: usize) -> Node {
                let ($($T,)*) = self;
                Node::from_iter([
                    $($T.replacement_node(span),)*
                ])
            }
        }
    }
}

opt_pattern!(A, B);
opt_pattern!(A, B, C);
opt_pattern!(A, B, C, D);
