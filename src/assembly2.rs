use std::{
    hash::{DefaultHasher, Hash, Hasher},
    ops::{Index, IndexMut},
};

use slotmap::SlotMap;

use crate::{
    function2::{FunctionFlags, FunctionId},
    Node, Signature,
};

/// A compiled Uiua assembly
#[derive(Clone, Default)]
pub struct Assembly {
    /// The top-level node
    pub root: Node,
    /// Functions
    functions: SlotMap<FunctionKeyInner, Node>,
}

slotmap::new_key_type! {
    struct FunctionKeyInner;
}

pub struct Function {
    inner: FunctionKeyInner,
    id: FunctionId,
    hash: u64,
    sig: Signature,
    flags: FunctionFlags,
}

#[derive(Debug, Clone)]
pub struct NewFunction {
    pub root: Node,
    pub flags: FunctionFlags,
}

impl From<Node> for NewFunction {
    fn from(root: Node) -> Self {
        Self {
            root,
            flags: FunctionFlags::default(),
        }
    }
}

impl Assembly {
    pub fn add_function(
        &mut self,
        id: FunctionId,
        sig: Signature,
        root: impl Into<NewFunction>,
    ) -> Function {
        let new_func = root.into();
        let mut hasher = DefaultHasher::new();
        new_func.root.hash(&mut hasher);
        let hash = hasher.finish();
        let inner = self.functions.insert(new_func.root);
        Function {
            inner,
            id,
            hash,
            sig,
            flags: new_func.flags,
        }
    }
}

impl Index<&Function> for Assembly {
    type Output = Node;
    #[track_caller]
    fn index(&self, func: &Function) -> &Self::Output {
        match self.functions.get(func.inner) {
            Some(node) => node,
            None => panic!("{}({:?}) not found in assembly", func.id, func.inner.0),
        }
    }
}

impl IndexMut<&Function> for Assembly {
    #[track_caller]
    fn index_mut(&mut self, func: &Function) -> &mut Self::Output {
        match self.functions.get_mut(func.inner) {
            Some(node) => node,
            None => panic!("{}({:?}) not found in assembly", func.id, func.inner.0),
        }
    }
}
