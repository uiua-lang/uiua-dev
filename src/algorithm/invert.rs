use std::{error::Error, fmt};

use crate::{
    assembly::{Assembly, Function},
    check::SigCheckError,
    FunctionId, Node, Primitive, Signature,
};

pub(crate) const DEBUG: bool = false;

macro_rules! dbgln {
    ($($arg:tt)*) => {
        if DEBUG {
            println!($($arg)*); // Allow println
        }
    }
}

impl Node {
    pub fn un_inverse(&self, asm: &Assembly) -> InversionResult<Node> {
        todo!("new un inversion")
    }
    pub fn anti_inverse(&self, asm: &Assembly) -> InversionResult<Node> {
        todo!("new anti inversion")
    }
    pub fn under_inverse(&self, g_sig: Signature, asm: &Assembly) -> InversionResult<(Node, Node)> {
        todo!("new under inversion")
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum InversionError {
    #[default]
    Generic,
    TooManyInstructions,
    Signature(SigCheckError),
    InnerFunc(Vec<FunctionId>, Box<Self>),
    AsymmetricUnderSig(Signature),
    ComplexInvertedUnder,
    UnderExperimental,
}

type InversionResult<T = ()> = Result<T, InversionError>;

impl fmt::Display for InversionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InversionError::Generic => write!(f, "No inverse found"),
            InversionError::TooManyInstructions => {
                write!(f, "Function has too many instructions to invert")
            }
            InversionError::Signature(e) => write!(f, "Cannot invert invalid signature: {e}"),
            InversionError::InnerFunc(ids, inner) => {
                write!(f, "Inversion failed:")?;
                for id in ids {
                    write!(f, " cannot invert {id} because")?;
                }
                let inner = inner.to_string().to_lowercase();
                write!(f, " {inner}")
            }
            InversionError::AsymmetricUnderSig(sig) => {
                write!(
                    f,
                    "Cannot invert under with asymmetric \
                    second function signature {sig}"
                )
            }
            InversionError::ComplexInvertedUnder => {
                write!(f, "This under itself is too complex to invert")
            }
            InversionError::UnderExperimental => {
                write!(
                    f,
                    "Inversion of {} is experimental. To enable it, \
                    add `# Experimental!` to the top of the file.",
                    Primitive::Under.format()
                )
            }
        }
    }
}

impl InversionError {
    fn func(self, f: &Function) -> Self {
        match self {
            InversionError::InnerFunc(mut ids, inner) => {
                ids.push(f.id.clone());
                InversionError::InnerFunc(ids, inner)
            }
            e => InversionError::InnerFunc(vec![f.id.clone()], Box::new(e)),
        }
    }
}

impl From<SigCheckError> for InversionError {
    fn from(e: SigCheckError) -> Self {
        InversionError::Signature(e)
    }
}
impl From<()> for InversionError {
    fn from(_: ()) -> Self {
        InversionError::Generic
    }
}

impl Error for InversionError {}

use InversionError::Generic;
fn generic<T>() -> InversionResult<T> {
    Err(InversionError::Generic)
}
