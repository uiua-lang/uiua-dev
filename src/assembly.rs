use std::{
    fmt,
    hash::{DefaultHasher, Hash, Hasher},
    ops::{Index, IndexMut},
    path::PathBuf,
    str::FromStr,
    sync::Arc,
};

use dashmap::DashMap;
use ecow::{eco_vec, EcoString, EcoVec};
use serde::*;
use slotmap::SlotMap;

use crate::{
    compile::{LocalName, Module},
    function::FunctionFlags,
    is_ident_char, CodeSpan, FunctionId, InputSrc, IntoInputSrc, Node, SigNode, Signature, Span,
    Uiua, UiuaResult, Value,
};

/// A compiled Uiua assembly
#[derive(Clone)]
pub struct Assembly {
    /// The top-level node
    pub root: Node,
    /// Functions
    functions: SlotMap<FunctionKeyInner, Node>,
    /// A list of global bindings
    pub bindings: EcoVec<BindingInfo>,
    pub(crate) spans: EcoVec<Span>,
    pub(crate) inputs: Inputs,
    pub(crate) dynamic_functions: EcoVec<DynFn>,
}

slotmap::new_key_type! {
    struct FunctionKeyInner;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    inner: FunctionKeyInner,
    pub id: FunctionId,
    hash: u64,
    sig: Signature,
    pub flags: FunctionFlags,
}

impl PartialEq for Function {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.sig == other.sig && self.hash == other.hash
    }
}

impl Eq for Function {}

impl Hash for Function {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}

impl Function {
    pub fn sig(&self) -> Signature {
        self.sig
    }
}

impl Assembly {
    pub fn sig_node(&self, f: &Function) -> SigNode {
        SigNode::new(self[f].clone(), f.sig)
    }
    pub fn add_function(&mut self, id: FunctionId, sig: Signature, root: Node) -> Function {
        let mut hasher = DefaultHasher::new();
        root.hash(&mut hasher);
        let hash = hasher.finish();
        let inner = self.functions.insert(root);
        Function {
            inner,
            id,
            hash,
            sig,
            flags: FunctionFlags::default(),
        }
    }
    pub(crate) fn add_binding_at(
        &mut self,
        local: LocalName,
        global: BindingKind,
        span: Option<CodeSpan>,
        comment: Option<DocComment>,
    ) {
        let binding = BindingInfo {
            public: local.public,
            kind: global,
            span: span.unwrap_or_else(CodeSpan::dummy),
            comment,
        };
        if local.index < self.bindings.len() {
            self.bindings.make_mut()[local.index] = binding;
        } else {
            while self.bindings.len() < local.index {
                self.bindings.push(BindingInfo {
                    kind: BindingKind::Const(None),
                    public: false,
                    span: CodeSpan::dummy(),
                    comment: None,
                });
            }
            self.bindings.push(binding);
        }
    }
    pub(crate) fn bind_function(
        &mut self,
        local: LocalName,
        function: Function,
        span: usize,
        comment: Option<DocComment>,
    ) {
        let span = self.spans[span].clone();
        self.add_binding_at(local, BindingKind::Func(function), span.code(), comment);
    }
    pub(crate) fn bind_const(
        &mut self,
        local: LocalName,
        value: Option<Value>,
        span: usize,
        comment: Option<DocComment>,
    ) {
        let span = self.spans[span].clone();
        self.add_binding_at(local, BindingKind::Const(value), span.code(), comment);
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

type DynFn = Arc<dyn Fn(&mut Uiua) -> UiuaResult + Send + Sync + 'static>;

impl Default for Assembly {
    fn default() -> Self {
        Self {
            root: Node::default(),
            functions: SlotMap::default(),
            spans: eco_vec![Span::Builtin],
            bindings: EcoVec::new(),
            dynamic_functions: EcoVec::new(),
            inputs: Inputs::default(),
        }
    }
}

impl From<&Assembly> for Assembly {
    fn from(asm: &Assembly) -> Self {
        asm.clone()
    }
}

/// Information about a binding
#[derive(Debug, Clone)]
pub struct BindingInfo {
    /// The binding kind
    pub kind: BindingKind,
    /// Whether the binding is public
    pub public: bool,
    /// The span of the original binding name
    pub span: CodeSpan,
    /// The comment preceding the binding
    pub comment: Option<DocComment>,
}

/// A kind of global binding
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BindingKind {
    /// A constant value
    Const(Option<Value>),
    /// A function
    Func(Function),
    /// An imported module
    Import(PathBuf),
    /// A scoped module
    Module(Module),
    /// An index macro
    ///
    /// Contains the number of arguments
    IndexMacro(usize),
    /// A code macro
    CodeMacro(Node),
}

impl BindingKind {
    /// Get the signature of the binding
    pub fn signature(&self) -> Option<Signature> {
        match self {
            Self::Const(_) => Some(Signature::new(0, 1)),
            Self::Func(func) => Some(func.sig()),
            Self::Import { .. } => None,
            Self::Module(_) => None,
            Self::IndexMacro(_) => None,
            Self::CodeMacro(_) => None,
        }
    }
    /// Check if the global is a once-bound constant
    pub fn is_constant(&self) -> bool {
        matches!(self, Self::Const(_))
    }
}

/// A comment that documents a binding
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct DocComment {
    /// The comment text
    pub text: EcoString,
    /// The signature of the binding
    pub sig: Option<DocCommentSig>,
}

/// A signature in a doc comment
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct DocCommentSig {
    /// The arguments of the signature
    pub args: Vec<DocCommentArg>,
    /// The outputs of the signature
    pub outputs: Option<Vec<DocCommentArg>>,
}

impl DocCommentSig {
    /// Whether the doc comment signature matches a given function signature
    pub fn matches_sig(&self, sig: Signature) -> bool {
        self.args.len() == sig.args
            && (self.outputs.as_ref()).map_or(true, |o| o.len() == sig.outputs)
    }
    pub(crate) fn sig_string(&self) -> String {
        if let Some(outputs) = &self.outputs {
            format!(
                "signature {}",
                Signature::new(self.args.len(), outputs.len())
            )
        } else {
            format!("{} args", self.args.len())
        }
    }
}

impl fmt::Display for DocCommentSig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(outputs) = &self.outputs {
            for output in outputs {
                write!(f, " {}", output.name)?;
                if let Some(ty) = &output.ty {
                    write!(f, ":{}", ty)?;
                }
            }
            write!(f, " ")?;
        }
        write!(f, "? ")?;
        for (i, arg) in self.args.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{}", arg.name)?;
            if let Some(ty) = &arg.ty {
                write!(f, ":{}", ty)?;
            }
        }
        Ok(())
    }
}

/// An argument in a doc comment signature
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct DocCommentArg {
    /// The name of the argument
    pub name: EcoString,
    /// A type descriptor for the argument
    pub ty: Option<EcoString>,
}

impl FromStr for DocCommentSig {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.trim_end().ends_with('?') && !s.trim_end().ends_with(" ?")
            || !(s.chars()).all(|c| c.is_whitespace() || "?:".contains(c) || is_ident_char(c))
        {
            return Err(());
        }
        // Split into args and outputs
        let (mut outputs_text, mut args_text) = s.split_once('?').ok_or(())?;
        outputs_text = outputs_text.trim();
        args_text = args_text.trim();
        // Parse args and outputs
        let mut args = Vec::new();
        let mut outputs = Vec::new();
        for (args, text) in [(&mut args, args_text), (&mut outputs, outputs_text)] {
            // Tokenize text
            let mut tokens = Vec::new();
            for frag in text.split_whitespace() {
                for (i, token) in frag.split(':').enumerate() {
                    if i > 0 {
                        tokens.push(":");
                    }
                    tokens.push(token);
                }
            }
            // Parse tokens into args
            let mut curr_arg_name = None;
            let mut tokens = tokens.into_iter().peekable();
            while let Some(token) = tokens.next() {
                if token == ":" {
                    let ty = tokens.next().unwrap_or_default();
                    args.push(DocCommentArg {
                        name: curr_arg_name.take().unwrap_or_default(),
                        ty: if ty.is_empty() { None } else { Some(ty.into()) },
                    });
                } else {
                    if let Some(curr) = curr_arg_name.take() {
                        args.push(DocCommentArg {
                            name: curr,
                            ty: None,
                        });
                    }
                    curr_arg_name = Some(token.into());
                }
            }
            if let Some(curr) = curr_arg_name.take() {
                args.push(DocCommentArg {
                    name: curr,
                    ty: None,
                });
            }
        }
        Ok(DocCommentSig {
            args,
            outputs: (!outputs.is_empty()).then_some(outputs),
        })
    }
}

impl From<String> for DocComment {
    fn from(text: String) -> Self {
        Self::from(text.as_str())
    }
}

impl From<&str> for DocComment {
    fn from(text: &str) -> Self {
        let mut sig = None;
        let sig_line = text.lines().position(|line| {
            line.chars().filter(|&c| c == '?').count() == 1
                && !line.trim().ends_with('?')
                && (line.chars()).all(|c| c.is_whitespace() || "?:".contains(c) || is_ident_char(c))
        });
        let raw_text = if let Some(i) = sig_line {
            sig = text.lines().nth(i).unwrap().parse().ok();

            let mut text: EcoString = (text.lines().take(i))
                .chain(["\n"])
                .chain(text.lines().skip(i + 1))
                .flat_map(|s| s.chars().chain(Some('\n')))
                .collect();
            while text.ends_with('\n') {
                text.pop();
            }
            if text.starts_with('\n') {
                text = text.trim_start_matches('\n').into();
            }
            text
        } else {
            text.into()
        };
        let mut text = EcoString::new();
        for (i, line) in raw_text.lines().enumerate() {
            if i > 0 {
                text.push('\n');
            }
            text.push_str(line.trim());
        }
        DocComment { text, sig }
    }
}

/// A repository of code strings input to the compiler
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Inputs {
    /// A map of file paths to their string contents
    #[serde(skip_serializing_if = "DashMap::is_empty")]
    pub files: DashMap<PathBuf, EcoString>,
    /// A list of input strings without paths
    #[serde(skip_serializing_if = "EcoVec::is_empty")]
    pub strings: EcoVec<EcoString>,
    /// A map of spans to macro strings
    #[serde(skip)]
    pub macros: DashMap<CodeSpan, EcoString>,
}

impl Inputs {
    pub(crate) fn add_src(
        &mut self,
        src: impl IntoInputSrc,
        input: impl Into<EcoString>,
    ) -> InputSrc {
        let src = src.into_input_src(self.strings.len());
        match &src {
            InputSrc::File(path) => {
                self.files.insert(path.to_path_buf(), input.into());
            }
            InputSrc::Str(i) => {
                while self.strings.len() <= *i {
                    self.strings.push(EcoString::default());
                }
                self.strings.make_mut()[*i] = input.into();
            }
            InputSrc::Macro(span) => {
                self.macros.insert((**span).clone(), input.into());
            }
        }
        src
    }
    /// Get an input string
    pub fn get(&self, src: &InputSrc) -> EcoString {
        match src {
            InputSrc::File(path) => self
                .files
                .get(&**path)
                .unwrap_or_else(|| panic!("File {:?} not found", path))
                .clone(),
            InputSrc::Str(index) => self
                .strings
                .get(*index)
                .unwrap_or_else(|| panic!("String {} not found", index))
                .clone(),
            InputSrc::Macro(span) => self
                .macros
                .get(span)
                .unwrap_or_else(|| panic!("Macro at {} not found", span))
                .clone(),
        }
    }
    /// Get an input string and perform an operation on it
    pub fn get_with<T>(&self, src: &InputSrc, f: impl FnOnce(&str) -> T) -> T {
        match src {
            InputSrc::File(path) => {
                if let Some(src) = self.files.get(&**path) {
                    f(&src)
                } else {
                    panic!("File {:?} not found", path)
                }
            }
            InputSrc::Str(index) => {
                if let Some(src) = self.strings.get(*index) {
                    f(src)
                } else {
                    panic!("String {} not found", index)
                }
            }
            InputSrc::Macro(span) => {
                if let Some(src) = self.macros.get(span) {
                    f(src.value())
                } else {
                    panic!("Macro at {} not found", span)
                }
            }
        }
    }
    /// Get an input string and perform an operation on it
    pub fn try_get_with<T>(&self, src: &InputSrc, f: impl FnOnce(&str) -> T) -> Option<T> {
        match src {
            InputSrc::File(path) => self.files.get(&**path).map(|src| f(&src)),
            InputSrc::Str(index) => self.strings.get(*index).map(|src| f(src)),
            InputSrc::Macro(span) => self.macros.get(span).map(|src| f(&src)),
        }
    }
}
