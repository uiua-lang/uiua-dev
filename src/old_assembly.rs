use std::{fmt, path::PathBuf, str::FromStr, sync::Arc};

use dashmap::DashMap;
use ecow::{eco_vec, EcoString, EcoVec};
use serde::*;

use crate::{
    is_ident_char, CodeSpan, FuncSlice, Function, InputSrc, Instr, IntoInputSrc, LocalName, Module,
    Signature, Span, Uiua, UiuaResult, Value,
};

/// A compiled Uiua assembly
#[derive(Clone)]
pub struct Assembly {
    pub(crate) instrs: EcoVec<Instr>,
    /// The sections of the instructions that are top-level expressions
    pub(crate) top_slices: Vec<FuncSlice>,
    /// A list of global bindings
    pub bindings: EcoVec<BindingInfo>,
    pub(crate) spans: EcoVec<Span>,
    pub(crate) inputs: Inputs,
    pub(crate) dynamic_functions: EcoVec<DynFn>,
}

type DynFn = Arc<dyn Fn(&mut Uiua) -> UiuaResult + Send + Sync + 'static>;

impl Default for Assembly {
    fn default() -> Self {
        Self {
            instrs: EcoVec::new(),
            top_slices: Vec::new(),
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

impl Assembly {
    /// Get the instructions of a function slice
    #[track_caller]
    pub fn instrs(&self, slice: FuncSlice) -> &[Instr] {
        let end = slice.end();
        assert!(
            slice.start <= self.instrs.len(),
            "Func slice start {} out of bounds of {} instrs",
            slice.start,
            self.instrs.len()
        );
        assert!(
            end <= self.instrs.len(),
            "Func slice end {} out of bounds of {} instrs",
            end,
            self.instrs.len()
        );
        &self.instrs[slice.start..end]
    }
    /// Get the mutable instructions of a function slice
    pub fn instrs_mut(&mut self, slice: FuncSlice) -> &mut [Instr] {
        &mut self.instrs.make_mut()[slice.start..slice.end()]
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
    /// Remove dead code from the assembly
    pub fn remove_dead_code(&mut self) {
        let mut slices = Vec::new();
        let mut bindings = Vec::new();
        for (i, binding) in self.bindings.iter().enumerate() {
            if let BindingKind::CodeMacro(mut slice) = binding.kind {
                if slice.start > 0 {
                    if let Some((Instr::Comment(before), Instr::Comment(after))) = self
                        .instrs
                        .get(slice.start - 1)
                        .zip(self.instrs.get(slice.end()))
                    {
                        if before.starts_with('(') && after.ends_with(')') {
                            slice.start -= 1;
                            slice.len += 2;
                        }
                    }
                }
                slices.push(slice);
                bindings.push(i);
            }
        }
        for i in bindings.into_iter().rev() {
            self.remove_binding(i);
        }
        slices.sort();
        for slice in slices.into_iter().rev() {
            self.remove_slice(slice);
        }
    }
    fn remove_binding(&mut self, i: usize) -> BindingInfo {
        for instr in self.instrs.make_mut() {
            match instr {
                Instr::CallGlobal { index, .. } | Instr::BindGlobal { index, .. } if *index > i => {
                    *index -= 1
                }
                _ => {}
            }
        }
        self.bindings.remove(i)
    }
    pub(crate) fn remove_slice(&mut self, rem_slice: FuncSlice) -> Vec<Instr> {
        // Remove instrs
        let after: Vec<_> = self.instrs[rem_slice.end()..].to_vec();
        let removed = self.instrs[rem_slice.start..rem_slice.end()].to_vec();
        // println!(
        //     "remove {}-{}: {:?}",
        //     rem_slice.start,
        //     rem_slice.end(),
        //     removed
        // );
        self.instrs.truncate(rem_slice.start);
        self.instrs.extend(after);
        // Remove top slices
        for slice in &mut self.top_slices {
            if slice.start > rem_slice.start {
                slice.start -= rem_slice.len();
            }
        }
        // Decrement bindings
        for binding in self.bindings.make_mut() {
            match &mut binding.kind {
                BindingKind::Func(func) => {
                    if func.slice.start > rem_slice.start {
                        func.slice.start -= rem_slice.len();
                    }
                }
                BindingKind::CodeMacro(slice) => {
                    if slice.start > rem_slice.start {
                        slice.start -= rem_slice.len();
                    }
                }
                _ => (),
            }
        }
        // Decrement instrs
        for instr in self.instrs.make_mut() {
            if let Instr::PushFunc(func) = instr {
                if func.slice.start > rem_slice.start {
                    func.slice.start -= rem_slice.len();
                }
            }
        }

        removed
    }
    /// Make top-level expressions not run
    pub fn remove_top_level(&mut self) {
        self.top_slices.clear();
    }
    /// Parse a `.uasm` file into an assembly
    pub fn from_uasm(src: &str) -> Result<Self, String> {
        let rest = src;
        let (instrs_src, rest) = rest
            .trim()
            .split_once("TOP SLICES")
            .ok_or("No top slices")?;
        let (top_slices_src, rest) = rest.split_once("BINDINGS").ok_or("No bindings")?;
        let (bindings_src, rest) = rest.trim().split_once("SPANS").ok_or("No spans")?;
        let (spans_src, rest) = rest.trim().split_once("FILES").ok_or("No files")?;
        let (files_src, rest) = rest
            .trim()
            .split_once("STRING INPUTS")
            .unwrap_or((rest, ""));
        let strings_src = rest.trim();

        let mut instrs = EcoVec::new();
        for line in instrs_src.lines().filter(|line| !line.trim().is_empty()) {
            let instr: Instr = serde_json::from_str(line)
                .or_else(|e| {
                    let (key, val) = line.split_once(' ').ok_or("No key")?;
                    let json = format!("{{{key:?}: {val}}}");
                    serde_json::from_str(&json).map_err(|_| e.to_string())
                })
                .or_else(|e| {
                    let (key, val) = line.split_once(' ').ok_or("No key")?;
                    let json = format!("[{key:?},{val}]");
                    serde_json::from_str(&json).map_err(|_| e)
                })
                .or_else(|e| serde_json::from_str(&format!("\"{line}\"")).map_err(|_| e))
                .unwrap();
            instrs.push(instr);
        }

        let mut top_slices = Vec::new();
        for line in top_slices_src
            .lines()
            .filter(|line| !line.trim().is_empty())
        {
            let (start, len) = line.split_once(' ').ok_or("No start")?;
            let start = start.parse::<usize>().map_err(|e| e.to_string())?;
            let len = len.parse::<usize>().map_err(|e| e.to_string())?;
            top_slices.push(FuncSlice { start, len });
        }

        let mut bindings = EcoVec::new();
        for line in bindings_src.lines().filter(|line| !line.trim().is_empty()) {
            let (public, line) = if let Some(line) = line.strip_prefix("private ") {
                (false, line)
            } else {
                (true, line)
            };
            let kind: BindingKind = serde_json::from_str(line).or_else(|e| {
                if let Some((key, val)) = line.split_once("\" ") {
                    let key = format!("{key}\"");
                    let json = format!("{{{key:?}: {val:?}}}");
                    serde_json::from_str(&json).map_err(|_| e.to_string())
                } else if let Some((key, val)) = line.split_once(' ') {
                    let json = format!("{{{key:?}: {val}}}");
                    serde_json::from_str(&json).map_err(|_| e.to_string())
                } else {
                    Err("No key".into())
                }
            })?;
            bindings.push(BindingInfo {
                kind,
                public,
                span: CodeSpan::dummy(),
                comment: None,
            });
        }

        let mut spans = EcoVec::new();
        spans.push(Span::Builtin);
        for line in spans_src.lines().filter(|line| !line.trim().is_empty()) {
            if line.trim().is_empty() {
                spans.push(Span::Builtin);
            } else {
                let (src_start, end) = line.trim().rsplit_once(' ').ok_or("invalid span")?;
                let (src, start) = src_start.split_once(' ').ok_or("invalid span")?;
                let src = serde_json::from_str(src).map_err(|e| e.to_string())?;
                let start = serde_json::from_str(start).map_err(|e| e.to_string())?;
                let end = serde_json::from_str(end).map_err(|e| e.to_string())?;
                spans.push(Span::Code(CodeSpan { src, start, end }));
            }
        }

        let files = DashMap::new();
        for line in files_src.lines().filter(|line| !line.trim().is_empty()) {
            let (path, src) = line.split_once(": ").ok_or("No path")?;
            let path = PathBuf::from(path);
            let src: EcoString = serde_json::from_str(src).map_err(|e| e.to_string())?;
            files.insert(path, src);
        }

        let mut strings = EcoVec::new();
        for line in strings_src.lines() {
            let src: EcoString = serde_json::from_str(line).map_err(|e| e.to_string())?;
            strings.push(src);
        }

        Ok(Self {
            instrs,
            top_slices,
            bindings,
            spans,
            inputs: Inputs {
                files,
                strings,
                ..Inputs::default()
            },
            dynamic_functions: EcoVec::new(),
        })
    }
    /// Serialize the assembly into a `.uasm` file
    pub fn to_uasm(&self) -> String {
        let mut uasm = String::new();

        for instr in &self.instrs {
            let json = serde_json::to_value(instr).unwrap();
            match &json {
                serde_json::Value::Object(map) => {
                    if map.len() == 1 {
                        let key = map.keys().next().unwrap();
                        let value = map.values().next().unwrap();
                        uasm.push_str(&format!("{} {}\n", key, value));
                        continue;
                    }
                }
                serde_json::Value::Array(arr) => {
                    if arr.len() == 2 {
                        if let serde_json::Value::String(key) = &arr[0] {
                            let value = &arr[1];
                            uasm.push_str(&format!("{} {}\n", key, value));
                            continue;
                        }
                    }
                }
                serde_json::Value::String(s) => {
                    uasm.push_str(&format!("{s:?}"));
                    uasm.push('\n');
                    continue;
                }
                _ => (),
            }
            uasm.push_str(&json.to_string());
            uasm.push('\n');
        }

        uasm.push_str("\nTOP SLICES\n");
        for slice in &self.top_slices {
            uasm.push_str(&format!("{} {}\n", slice.start, slice.len));
        }

        uasm.push_str("\nBINDINGS\n");
        for binding in &self.bindings {
            if !binding.public {
                uasm.push_str("private ");
            }
            if let serde_json::Value::Object(map) = serde_json::to_value(&binding.kind).unwrap() {
                if map.len() == 1 {
                    let key = map.keys().next().unwrap();
                    let value = map.values().next().unwrap();
                    uasm.push_str(&format!("{} {}\n", key, value));
                    continue;
                }
            }
            uasm.push_str(&serde_json::to_string(&binding.kind).unwrap());
            uasm.push('\n');
        }

        uasm.push_str("\nSPANS\n");
        for span in self.spans.iter().skip(1) {
            if let Span::Code(span) = span {
                uasm.push_str(&serde_json::to_string(&span.src).unwrap());
                uasm.push(' ');
                uasm.push_str(&serde_json::to_string(&span.start).unwrap());
                uasm.push(' ');
                uasm.push_str(&serde_json::to_string(&span.end).unwrap());
            }
            uasm.push('\n');
        }

        uasm.push_str("\nFILES\n");
        for entry in &self.inputs.files {
            let key = entry.key();
            let value = entry.value();
            uasm.push_str(&format!("{}: {:?}\n", key.display(), value));
        }

        if !self.inputs.strings.is_empty() {
            uasm.push_str("\nSTRING INPUTS\n");
            for src in &self.inputs.strings {
                uasm.push_str(&serde_json::to_string(src).unwrap());
                uasm.push('\n');
            }
        }

        uasm
    }
}

impl AsRef<Assembly> for Assembly {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl AsMut<Assembly> for Assembly {
    fn as_mut(&mut self) -> &mut Self {
        self
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
    CodeMacro(FuncSlice),
}

impl BindingKind {
    /// Get the signature of the binding
    pub fn signature(&self) -> Option<Signature> {
        match self {
            Self::Const(_) => Some(Signature::new(0, 1)),
            Self::Func(func) => Some(func.signature()),
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
