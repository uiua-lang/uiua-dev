use std::{
    fmt,
    hash::Hash,
    ops::{BitAnd, BitOr, BitOrAssign},
};

use serde::*;

use crate::Signature;

#[derive(Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct FunctionFlags(u8);

impl BitOrAssign for FunctionFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitOr for FunctionFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitAnd for FunctionFlags {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl FunctionFlags {
    pub const RECURSIVE: Self = Self(1 << 0);
    pub const NO_INLINE: Self = Self(1 << 1);
    pub const TRACK_CALLER: Self = Self(1 << 2);
    pub const NO_PRE_EVAL: Self = Self(1 << 3);
    pub fn recursive(&self) -> bool {
        self.0 & Self::RECURSIVE.0 != 0
    }
    pub fn no_inline(&self) -> bool {
        self.0 & Self::NO_INLINE.0 != 0
    }
    pub fn track_caller(&self) -> bool {
        self.0 & Self::TRACK_CALLER.0 != 0
    }
    pub fn no_pre_eval(&self) -> bool {
        self.0 & Self::NO_PRE_EVAL.0 != 0
    }
}

impl fmt::Debug for FunctionFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut wrote = false;
        for (i, name) in ["RECURSIVE", "NO_INLINE", "TRACK_CALLER", "NO_PRE_EVAL"]
            .into_iter()
            .enumerate()
        {
            if self.0 & (1 << i) != 0 {
                if wrote {
                    write!(f, " | ")?;
                }
                wrote = true;
                write!(f, "{name}")?;
            }
        }
        Ok(())
    }
}

/// A function that executes Rust code
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DynamicFunction {
    /// An index used to look up the function
    pub(crate) index: usize,
    /// The function's signature
    pub(crate) sig: Signature,
}

impl From<(usize, Signature)> for DynamicFunction {
    fn from((index, sig): (usize, Signature)) -> Self {
        Self { index, sig }
    }
}

impl From<DynamicFunction> for (usize, Signature) {
    fn from(func: DynamicFunction) -> Self {
        (func.index, func.sig)
    }
}

impl DynamicFunction {
    /// Get the function's signature
    pub fn signature(&self) -> Signature {
        self.sig
    }
}

impl fmt::Debug for DynamicFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<dynamic#{:x}>", self.index)
    }
}
