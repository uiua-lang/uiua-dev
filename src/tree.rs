use std::{
    collections::hash_map::DefaultHasher,
    fmt,
    hash::{Hash, Hasher},
    iter::once,
    mem::take,
    slice,
};

use ecow::EcoVec;
use serde::{Deserialize, Serialize};

use crate::{ImplPrimitive, Primitive, Value};

macro_rules! node {
    ($(
        $(#[$attr:meta])*
        $((#[$rep_attr:meta] rep),)?
        (
            $num:literal,
            $name:ident
            $(($($tup_name:ident($tup_type:ty)),* $(,)?))?
            $({$($field_name:ident : $field_type:ty),* $(,)?})?
        )
    ),* $(,)?) => {
        /// A Uiua bytecode instruction
        #[derive(Clone, Serialize, Deserialize)]
        #[repr(u8)]
        #[allow(missing_docs)]
        #[serde(from = "NodeRep", into = "NodeRep")]
        pub enum Node {
            $(
                $(#[$attr])*
                $name $(($($tup_type),*))? $({$($field_name : $field_type),*})? = $num,
            )*
        }

        macro_rules! field_span {
            (span, $sp:ident) => {
                return Some($sp)
            };
            ($sp:ident, $sp2:ident) => {};
        }

        impl Node {
            /// Get the span index of this instruction
            #[allow(unreachable_code, unused)]
            pub fn span(&self) -> Option<usize> {
                (|| match self {
                    $(
                        Self::$name $(($($tup_name),*))? $({$($field_name),*})? => {
                            $($(field_span!($tup_name, $tup_name);)*)*
                            $($(field_span!($field_name, $field_name);)*)*
                            return None;
                        },
                    )*
                })().copied()
            }
            /// Get a mutable reference to the span index of this instruction
            #[allow(unreachable_code, unused)]
            pub fn span_mut(&mut self) -> Option<&mut usize> {
                match self {
                    $(
                        Self::$name $(($($tup_name),*))? $({$($field_name),*})? => {
                            $($(field_span!($tup_name, $tup_name);)*)*
                            $($(field_span!($field_name, $field_name);)*)*
                            return None;
                        },
                    )*
                }
            }
        }

        impl PartialEq for Node {
            #[allow(unused_variables)]
            fn eq(&self, other: &Self) -> bool {
                let mut hasher = DefaultHasher::new();
                self.hash(&mut hasher);
                let hash = hasher.finish();
                let mut other_hasher = DefaultHasher::new();
                other.hash(&mut other_hasher);
                let other_hash = other_hasher.finish();
                hash == other_hash
            }
        }

        impl Eq for Node {}

        impl Hash for Node {
            #[allow(unused_variables)]
            fn hash<H: Hasher>(&self, state: &mut H) {
                macro_rules! hash_field {
                    (span) => {};
                    ($nm:ident) => {Hash::hash($nm, state)};
                }
                match self {
                    $(
                        Self::$name $(($($tup_name),*))? $({$($field_name),*})? => {
                            $num.hash(state);
                            $($(hash_field!($field_name);)*)?
                            $($(hash_field!($tup_name);)*)?
                        }
                    )*
                }
            }
        }

        #[derive(Serialize, Deserialize)]
        pub(crate) enum NodeRep {
            $(
                $(#[$rep_attr])?
                $name(
                    $($($tup_type),*)?
                    $($($field_type),*)?
                ),
            )*
        }

        impl From<NodeRep> for Node {
            fn from(rep: NodeRep) -> Self {
                match rep {
                    $(
                        NodeRep::$name (
                            $($($tup_name,)*)?
                            $($($field_name,)*)?
                        ) => Self::$name $(($($tup_name),*))? $({$($field_name),*})?,
                    )*
                }
            }
        }

        impl From<Node> for NodeRep {
            fn from(instr: Node) -> Self {
                match instr {
                    $(
                        Node::$name $(($($tup_name),*))? $({$($field_name),*})? => NodeRep::$name (
                            $($($tup_name),*)?
                            $($($field_name),*)?
                        ),
                    )*
                }
            }
        }
    };
}

node!(
    (0, Run(nodes(EcoVec<Node>))),
    (1, Push(val(Value))),
    (2, Prim(prim(Primitive), span(usize))),
    (3, ImplPrim(prim(ImplPrimitive), span(usize))),
    (4, Mod(prim(Primitive), args(EcoVec<Node>), span(usize))),
    (5, ImplMod(prim(ImplPrimitive), args(EcoVec<Node>), span(usize))),
    (6, CreateArray { len: usize, boxed: bool, span: usize }),
);

impl Default for Node {
    fn default() -> Self {
        Node::Run(EcoVec::new())
    }
}

impl Node {
    /// Get a slice of the nodes in this node
    pub fn as_slice(&self) -> &[Node] {
        if let Node::Run(nodes) = self {
            nodes
        } else {
            slice::from_ref(self)
        }
    }
    /// Get a mutable slice of the nodes in this node
    ///
    /// Transforms the node into a [`Node::Run`] if it is not already a [`Node::Run`]
    pub fn as_mut_slice(&mut self) -> &mut [Node] {
        match self {
            Node::Run(nodes) => nodes.make_mut(),
            other => {
                let first = take(other);
                let Node::Run(nodes) = other else {
                    unreachable!()
                };
                nodes.push(first);
                nodes.make_mut()
            }
        }
    }
    /// Get a mutable vector of the nodes in this node
    ///
    /// Transforms the node into a [`Node::Run`] if it is not already a [`Node::Run`]
    pub fn as_vec(&mut self) -> &mut EcoVec<Node> {
        match self {
            Node::Run(nodes) => nodes,
            other => {
                let first = take(other);
                let Node::Run(nodes) = other else {
                    unreachable!()
                };
                nodes.push(first);
                nodes
            }
        }
    }
    /// Push a node onto the end of the node
    ///
    /// Transforms the node into a [`Node::Run`] if it is not already a [`Node::Run`]
    pub fn push(&mut self, node: Node) {
        self.as_vec().push(node);
    }
    pub(crate) fn as_flipped_primitive(&self) -> Option<(Primitive, bool)> {
        match self {
            Node::Prim(prim, _) => Some((*prim, false)),
            Node::Run(nodes) => match nodes.as_slice() {
                [Node::Prim(Primitive::Flip, _), Node::Prim(prim, _)] => Some((*prim, true)),
                _ => None,
            },
            _ => None,
        }
    }
    pub(crate) fn as_primitive(&self) -> Option<Primitive> {
        self.as_flipped_primitive()
            .filter(|(_, flipped)| !flipped)
            .map(|(prim, _)| prim)
    }
    pub(crate) fn as_flipped_impl_primitive(&self) -> Option<(ImplPrimitive, bool)> {
        match self {
            Node::ImplPrim(prim, _) => Some((*prim, false)),
            Node::Run(nodes) => match nodes.as_slice() {
                [Node::Prim(Primitive::Flip, _), Node::ImplPrim(prim, _)] => Some((*prim, true)),
                _ => None,
            },
            _ => None,
        }
    }
    pub(crate) fn as_impl_primitive(&self) -> Option<ImplPrimitive> {
        self.as_flipped_impl_primitive()
            .filter(|(_, flipped)| !flipped)
            .map(|(prim, _)| prim)
    }
}

impl FromIterator<Node> for Node {
    fn from_iter<T: IntoIterator<Item = Node>>(iter: T) -> Self {
        let mut iter = iter.into_iter().peekable();
        let Some(first) = iter.next() else {
            return Node::default();
        };
        if iter.peek().is_none() {
            first
        } else {
            Node::Run(once(first).chain(iter).collect())
        }
    }
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Node::Run(eco_vec) => {
                let mut tuple = f.debug_tuple("");
                for node in eco_vec {
                    tuple.field(node);
                }
                tuple.finish()
            }
            Node::Push(value) => write!(f, "push {value}"),
            Node::Prim(prim, _) => write!(f, "{prim}"),
            Node::ImplPrim(impl_prim, _) => write!(f, "{impl_prim}"),
            Node::Mod(prim, args, _) => {
                let mut tuple = f.debug_tuple(&prim.to_string());
                for node in args {
                    tuple.field(node);
                }
                tuple.finish()
            }
            Node::ImplMod(impl_prim, args, _) => {
                let mut tuple = f.debug_tuple(&impl_prim.to_string());
                for node in args {
                    tuple.field(node);
                }
                tuple.finish()
            }
            Node::CreateArray {
                len, boxed: true, ..
            } => write!(f, "{{{}{}}}", Primitive::Len, len),
            Node::CreateArray {
                len, boxed: false, ..
            } => write!(f, "[{}{}]", Primitive::Len, len),
        }
    }
}
