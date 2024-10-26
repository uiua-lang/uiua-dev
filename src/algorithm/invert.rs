use std::{cell::RefCell, collections::HashMap, error::Error, fmt};

use crate::{
    assembly::{Assembly, Function},
    check::SigCheckError,
    FunctionId, Node, Primitive, Signature, Uiua, UiuaResult,
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

use ecow::{EcoString, EcoVec};
use regex::Regex;
use InversionError::Generic;
fn generic<T>() -> InversionResult<T> {
    Err(InversionError::Generic)
}

pub(crate) fn match_format_pattern(parts: EcoVec<EcoString>, env: &mut Uiua) -> UiuaResult {
    let val = env
        .pop(1)?
        .as_string(env, "Matching a format pattern expects a string")?;
    match parts.as_slice() {
        [] => {}
        [part] => {
            if val != part.as_ref() {
                return Err(env.error("String did not match pattern exactly"));
            }
        }
        _ => {
            thread_local! {
                static CACHE: RefCell<HashMap<EcoVec<EcoString>, Regex>> = RefCell::new(HashMap::new());
            }
            CACHE.with(|cache| {
                let mut cache = cache.borrow_mut();
                let re = cache.entry(parts.clone()).or_insert_with(|| {
                    let mut re = String::new();
                    re.push_str("(?s)^");
                    for (i, part) in parts.iter().enumerate() {
                        if i > 0 {
                            re.push_str("(.+?|.*)");
                        }
                        re.push_str(&regex::escape(part));
                    }
                    re.push('$');
                    Regex::new(&re).unwrap()
                });
                if !re.is_match(val.as_ref()) {
                    return Err(env.error("String did not match format string pattern"));
                }
                let captures = re.captures(val.as_ref()).unwrap();
                let caps: Vec<_> = captures.iter().skip(1).flatten().collect();
                for cap in caps.into_iter().rev() {
                    env.push(cap.as_str());
                }
                Ok(())
            })?;
        }
    }
    Ok(())
}
