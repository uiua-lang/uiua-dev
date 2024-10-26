mod binding;
mod data;
mod modifier;

use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::{BTreeSet, HashMap, HashSet, VecDeque},
    env::current_dir,
    fmt, fs,
    iter::{once, repeat},
    mem::{replace, swap, take},
    panic::{catch_unwind, AssertUnwindSafe},
    path::{Path, PathBuf},
    sync::Arc,
};

use ecow::{eco_vec, EcoString, EcoVec};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::{
    ast::*,
    check::{nodes_sig, SigCheckError, SigCheckErrorKind},
    format::{format_word, format_words},
    function::DynamicFunction,
    ident_modifier_args,
    lex::{CodeSpan, Sp, Span},
    lsp::{CodeMeta, ImportSrc, SigDecl},
    parse::{count_placeholders, flip_unsplit_lines, parse, split_words},
    Array, Assembly, BindingKind, Boxed, CustomInverse, Diagnostic, DiagnosticKind, DocComment,
    DocCommentSig, Function, FunctionId, GitTarget, Ident, ImplPrimitive, InputSrc, IntoInputSrc,
    IntoSysBackend, Node, Primitive, Purity, RunMode, SemanticComment, SigNode, Signature,
    SysBackend, Uiua, UiuaError, UiuaErrorKind, UiuaResult, Value, CONSTANTS, EXAMPLE_UA,
    SUBSCRIPT_NUMS, VERSION,
};

/// The Uiua compiler
#[derive(Clone)]
pub struct Compiler {
    pub(crate) asm: Assembly,
    pub(crate) code_meta: CodeMeta,
    /// The name of the current bindings
    current_bindings: Vec<CurrentBinding>,
    /// The index of the next global binding
    next_global: usize,
    /// The current scope
    pub(crate) scope: Scope,
    /// Ancestor scopes of the current one
    higher_scopes: Vec<Scope>,
    /// Determines which How test scopes are run
    mode: RunMode,
    /// The paths of files currently being imported (used to detect import cycles)
    current_imports: Vec<PathBuf>,
    /// The bindings of imported files
    imports: HashMap<PathBuf, Module>,
    /// Unexpanded index macros
    index_macros: HashMap<usize, IndexMacro>,
    /// Unexpanded code macros
    code_macros: HashMap<usize, CodeMacro>,
    /// The depth of macro expansion
    macro_depth: usize,
    /// Whether the compiler is in an inverse
    in_inverse: bool,
    /// Whether the compiler is in a try
    in_try: bool,
    /// Accumulated errors
    errors: Vec<UiuaError>,
    /// Primitives that have emitted errors because they are deprecated
    deprecated_prim_errors: HashSet<Primitive>,
    /// Accumulated diagnostics
    diagnostics: BTreeSet<Diagnostic>,
    /// Print diagnostics as they are encountered
    pub(crate) print_diagnostics: bool,
    /// Whether to evaluate comptime code
    comptime: bool,
    /// The comptime mode
    pre_eval_mode: PreEvalMode,
    /// The interpreter used for comptime code
    macro_env: Uiua,
}

impl Default for Compiler {
    fn default() -> Self {
        Compiler {
            asm: Assembly::default(),
            code_meta: CodeMeta::default(),
            current_bindings: Vec::new(),
            next_global: 0,
            scope: Scope::default(),
            higher_scopes: Vec::new(),
            mode: RunMode::All,
            current_imports: Vec::new(),
            imports: HashMap::new(),
            index_macros: HashMap::new(),
            code_macros: HashMap::new(),
            macro_depth: 0,
            in_inverse: false,
            in_try: false,
            errors: Vec::new(),
            deprecated_prim_errors: HashSet::new(),
            diagnostics: BTreeSet::new(),
            print_diagnostics: false,
            comptime: true,
            pre_eval_mode: PreEvalMode::default(),
            macro_env: Uiua::default(),
        }
    }
}

#[derive(Default)]
struct BindingPrelude {
    comment: Option<EcoString>,
}

/// A Uiua module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Module {
    /// The top level comment
    pub comment: Option<EcoString>,
    /// Map module-local names to global indices
    pub names: IndexMap<Ident, LocalName>,
    /// Whether the module uses experimental features
    experimental: bool,
}

/// An index macro
#[derive(Clone)]
struct IndexMacro {
    words: Vec<Sp<Word>>,
    names: IndexMap<Ident, LocalName>,
    sig: Option<Signature>,
    hygenic: bool,
    recursive: bool,
}

/// A code macro
#[derive(Clone)]
struct CodeMacro {
    root: SigNode,
    names: IndexMap<Ident, LocalName>,
}

impl AsRef<Assembly> for Compiler {
    fn as_ref(&self) -> &Assembly {
        &self.asm
    }
}

impl AsMut<Assembly> for Compiler {
    fn as_mut(&mut self) -> &mut Assembly {
        &mut self.asm
    }
}

#[derive(Clone)]
struct CurrentBinding {
    name: Ident,
    signature: Option<Signature>,
    referenced: bool,
    global_index: usize,
}

/// A scope where names are defined
#[derive(Clone)]
pub(crate) struct Scope {
    kind: ScopeKind,
    /// The name of the current file, if any
    file_path: Option<PathBuf>,
    /// The top level comment
    comment: Option<EcoString>,
    /// Map local names to global indices
    names: IndexMap<Ident, LocalName>,
    /// Number of named data variants
    data_variants: usize,
    /// Whether to allow experimental features
    pub experimental: bool,
    /// Whether an error has been emitted for experimental features
    experimental_error: bool,
    /// Whether an error has been emitted for fill function signatures
    fill_sig_error: bool,
    /// The stack height between top-level statements
    stack_height: Result<usize, Sp<SigCheckError>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ScopeKind {
    /// A scope at the top level of a file
    File(FileScopeKind),
    /// A scope in a named module
    Module(Ident),
    /// A scope that includes all bindings in a module
    AllInModule,
    /// A temporary scope, probably for a macro
    Temp(Option<MacroLocal>),
    /// A test scope between `---`s
    Test,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileScopeKind {
    Source,
    Git,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MacroLocal {
    macro_index: usize,
    expansion_index: usize,
}

impl Default for Scope {
    fn default() -> Self {
        Self {
            kind: ScopeKind::File(FileScopeKind::Source),
            file_path: None,
            comment: None,
            names: IndexMap::new(),
            data_variants: 0,
            experimental: false,
            experimental_error: false,
            fill_sig_error: false,
            stack_height: Ok(0),
        }
    }
}

/// The index of a named local in the bindings, and whether it is public
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LocalName {
    /// The index of the binding in assembly's bindings
    pub index: usize,
    /// Whether the binding is public
    pub public: bool,
}

/// The mode that dictates how much code to pre-evaluate at compile time
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum PreEvalMode {
    /// Evaluate as much as possible at compile time, even impure expressions
    ///
    /// Recursive functions and certain system functions are not evaluated
    Lsp,
    /// Does not evalute pure constants and expressions at comptime, but still evaluates `comptime`
    Lazy,
    /// Pre-evaluate each line, but not multiple lines together
    Line,
    /// The normal mode. Tries to evaluate pure, time-bounded constants and expressions at comptime
    #[default]
    Normal,
}

const MAX_PRE_EVAL_ELEMS: usize = 1000;
const MAX_PRE_EVAL_RANK: usize = 4;

impl PreEvalMode {
    #[allow(unused)]
    fn matches_node(&self, node: &Node, asm: &Assembly) -> bool {
        if node.iter().any(|node| {
            matches!(
                node,
                Node::Push(val)
                    if val.element_count() > MAX_PRE_EVAL_ELEMS
                    || val.rank() > MAX_PRE_EVAL_RANK
            )
        }) {
            return false;
        }
        match self {
            PreEvalMode::Normal | PreEvalMode::Line => {
                node.is_pure(Purity::Pure, asm) && node.is_limit_bounded(asm)
            }
            PreEvalMode::Lazy => false,
            PreEvalMode::Lsp => node.is_pure(Purity::Impure, asm) && node.is_limit_bounded(asm),
        }
    }
}

impl Compiler {
    /// Create a new compiler
    pub fn new() -> Self {
        Self::default()
    }
    /// Create a new compiler with a custom backend for `comptime` code
    pub fn with_backend(backend: impl IntoSysBackend) -> Self {
        Self {
            macro_env: Uiua::with_backend(backend.into_sys_backend()),
            ..Self::default()
        }
    }
    /// Set the compiler's assembly
    pub fn with_assembly(self, asm: Assembly) -> Self {
        Self { asm, ..self }
    }
    /// Get a reference to the assembly
    pub fn assembly(&self) -> &Assembly {
        &self.asm
    }
    /// Get a mutable reference to the assembly
    pub fn assembly_mut(&mut self) -> &mut Assembly {
        &mut self.asm
    }
    /// Get a reference to the code metadata
    pub fn code_meta(&self) -> &CodeMeta {
        &self.code_meta
    }
    /// Get a mutable reference to the code metadata
    pub fn code_meta_mut(&mut self) -> &mut CodeMeta {
        &mut self.code_meta
    }
    /// Take a completed assembly from the compiler
    pub fn finish(&mut self) -> Assembly {
        take(&mut self.asm)
    }
    /// Set whether to evaluate `comptime`
    pub fn comptime(&mut self, comptime: bool) -> &mut Self {
        self.comptime = comptime;
        self
    }
    /// Set the [`PreEvalMode`]
    pub fn pre_eval_mode(&mut self, mode: PreEvalMode) -> &mut Self {
        self.pre_eval_mode = mode;
        self
    }
    /// Set whether to print diagnostics as they are encountered
    ///
    /// If this is set to false, diagnostics will be accumulated and can be retrieved with [`Compiler::take_diagnostics`]
    ///
    /// Defaults to false
    pub fn print_diagnostics(&mut self, print_diagnostics: bool) -> &mut Self {
        self.print_diagnostics = print_diagnostics;
        self
    }
    /// Set the run mode
    pub fn mode(&mut self, mode: RunMode) -> &mut Self {
        self.mode = mode;
        self
    }
    /// Get the backend
    pub fn backend(&self) -> Arc<dyn SysBackend> {
        self.macro_env.rt.backend.clone()
    }
    /// Attempt to downcast the system backend to a concrete reference type
    pub fn downcast_backend<T: SysBackend>(&self) -> Option<&T> {
        self.macro_env.downcast_backend()
    }
    /// Attempt to downcast the system backend to a concrete mutable type
    pub fn downcast_backend_mut<T: SysBackend>(&mut self) -> Option<&mut T> {
        self.macro_env.downcast_backend_mut()
    }
    /// Take the system backend
    pub fn take_backend<T: SysBackend + Default>(&mut self) -> Option<T> {
        self.macro_env.take_backend()
    }
    /// Set the system backend
    pub fn set_backend<T: SysBackend>(&mut self, backend: T) {
        self.macro_env.rt.backend = Arc::new(backend);
    }
    /// Compile a Uiua file from a file at a path
    pub fn load_file<P: AsRef<Path>>(&mut self, path: P) -> UiuaResult<&mut Self> {
        let path = path.as_ref();
        let input: EcoString = fs::read_to_string(path)
            .map_err(|e| UiuaErrorKind::Load(path.into(), e.into()))?
            .into();
        // _ = crate::lsp::spans(&input);
        self.asm.inputs.files.insert(path.into(), input.clone());
        self.load_impl(&input, InputSrc::File(path.into()))
    }
    /// Compile a Uiua file from a string
    pub fn load_str(&mut self, input: &str) -> UiuaResult<&mut Self> {
        let src = self.asm.inputs.add_src((), input);
        self.load_impl(input, src)
    }
    /// Compile a Uiua file from a string with a path for error reporting
    pub fn load_str_src(&mut self, input: &str, src: impl IntoInputSrc) -> UiuaResult<&mut Self> {
        let src = self.asm.inputs.add_src(src, input);
        self.load_impl(input, src)
    }
    fn scopes(&self) -> impl Iterator<Item = &Scope> {
        once(&self.scope).chain(self.higher_scopes.iter().rev())
    }
    /// Run in a scoped context. Names defined in this context will be removed when the scope ends.
    ///
    /// While names defined in this context will be removed when the scope ends, values *bound* to
    /// those names will not.
    ///
    /// All other runtime state other than the stack, will also be restored.
    fn in_scope<T>(
        &mut self,
        kind: ScopeKind,
        f: impl FnOnce(&mut Self) -> UiuaResult<T>,
    ) -> UiuaResult<(Module, T)> {
        self.higher_scopes.push(take(&mut self.scope));
        self.scope.kind = kind;
        let res = f(self);
        let scope = replace(&mut self.scope, self.higher_scopes.pop().unwrap());
        let res = res?;
        let module = Module {
            comment: scope.comment,
            names: scope.names,
            experimental: scope.experimental,
        };
        Ok((module, res))
    }
    fn load_impl(&mut self, input: &str, src: InputSrc) -> UiuaResult<&mut Self> {
        let node_start = self.asm.root.len();
        let (items, errors, diagnostics) = parse(input, src.clone(), &mut self.asm.inputs);
        for diagnostic in diagnostics {
            self.emit_diagnostic_impl(diagnostic);
        }
        if !errors.is_empty() {
            return Err(UiuaErrorKind::Parse(errors, self.asm.inputs.clone().into()).into());
        }
        if let InputSrc::File(path) = &src {
            self.current_imports.push(path.to_path_buf());
            self.scope.file_path = Some(if path.is_absolute() {
                current_dir()
                    .ok()
                    .and_then(|dir| pathdiff::diff_paths(path, dir))
                    .unwrap_or_else(|| path.to_path_buf())
            } else {
                path.to_path_buf()
            });
        }

        let res = self.catching_crash(input, |env| env.items(items, false));

        if self.print_diagnostics {
            for diagnostic in self.take_diagnostics() {
                eprintln!("{}", diagnostic.report());
            }
        }

        if let InputSrc::File(_) = &src {
            self.current_imports.pop();
        }
        match res {
            Err(e) | Ok(Err(e)) => {
                self.asm.root.truncate(node_start);
                self.errors.push(e);
            }
            _ => {}
        }
        match self.errors.len() {
            0 => Ok(self),
            1 => Err(self.errors.pop().unwrap()),
            _ => Err(UiuaError::from_multi(take(&mut self.errors))),
        }
    }
    fn catching_crash<T>(
        &mut self,
        input: impl fmt::Display,
        f: impl FnOnce(&mut Self) -> T,
    ) -> UiuaResult<T> {
        match catch_unwind(AssertUnwindSafe(|| f(self))) {
            Ok(res) => Ok(res),
            Err(_) => Err(UiuaErrorKind::CompilerPanic(format!(
                "\
The compiler has crashed!
Hooray! You found a bug!
Please report this at http://github.com/uiua-lang/uiua/issues/new \
or on Discord at https://discord.gg/9CU2ME4kmn.

Uiua version {VERSION}

code:
{input}"
            ))
            .into()),
        }
    }
    pub(crate) fn items(&mut self, items: Vec<Item>, must_run: bool) -> UiuaResult {
        // Set scope comment
        if let Some(Item::Words(lines)) = items.first() {
            let mut started = false;
            let mut comment = String::new();
            for line in lines {
                for word in line {
                    match &word.value {
                        Word::Comment(c) => {
                            let mut c = c.as_str();
                            if c.starts_with(' ') {
                                c = &c[1..];
                            }
                            comment.push_str(c);
                            started = true;
                        }
                        Word::Spaces => {}
                        _ => {
                            comment.clear();
                            break;
                        }
                    }
                }
                if line.is_empty() && started {
                    break;
                }
                comment.push('\n');
            }
            if !comment.trim().is_empty() {
                self.scope.comment = Some(comment.trim().into());
            }
        }

        let mut prelude = BindingPrelude::default();
        let mut item_errored = false;
        let mut items = VecDeque::from(items);
        while let Some(item) = items.pop_front() {
            let must_run = must_run
                || matches!(&item, Item::Words(_))
                    && items.iter().any(|item| match item {
                        Item::Binding(binding)
                            if (binding.words.iter())
                                .filter(|word| word.value.is_code())
                                .count()
                                == 0 =>
                        {
                            true
                        }
                        Item::Words(lines) => lines.iter().any(|line| {
                            line.iter().find(|w| w.value.is_code()).is_some_and(|w| {
                                matches!(w.value, Word::Primitive(Primitive::Assert))
                            })
                        }),
                        _ => false,
                    });
            if let Err(e) = self.item(item, must_run, &mut prelude) {
                if !item_errored {
                    self.errors.push(e);
                }
                item_errored = true;
            }
        }
        Ok(())
    }
    fn item(&mut self, item: Item, must_run: bool, prelude: &mut BindingPrelude) -> UiuaResult {
        match item {
            Item::Module(m) => self.module(m, take(prelude).comment),
            Item::Words(lines) => self.lines(lines, must_run, true, prelude),
            Item::Binding(binding) => self.binding(binding, take(prelude)),
            Item::Import(import) => self.import(import, take(prelude).comment),
            Item::Data(data) => self.data_def(data, true, take(prelude)),
        }
    }
    fn lines(
        &mut self,
        mut lines: Vec<Vec<Sp<Word>>>,
        must_run: bool,
        _precomp: bool,
        prelude: &mut BindingPrelude,
    ) -> UiuaResult {
        fn words_should_run_anyway(words: &[Sp<Word>]) -> bool {
            let mut anyway = false;
            recurse_words(words, &mut |w| {
                anyway = anyway
                    || matches!(&w.value, Word::SemanticComment(_))
                    || matches!(&w.value, Word::Modified(m)
                        if matches!(m.modifier.value, Modifier::Ref(_)))
            });
            anyway
        }

        // Populate prelude
        for line in &mut lines {
            let mut words = line.iter().filter(|w| !matches!(w.value, Word::Spaces));
            if words.clone().count() == 1 {
                let word = words.next().unwrap();
                match &word.value {
                    Word::Comment(c) => {
                        if let Some(curr_com) = &mut prelude.comment {
                            curr_com.push('\n');
                            curr_com.push_str(c);
                        } else {
                            prelude.comment = Some(c.as_str().into());
                        }
                        line.clear();
                    }
                    Word::SemanticComment(SemanticComment::NoInline) => {
                        todo!()
                    }
                    Word::SemanticComment(SemanticComment::TrackCaller) => {
                        todo!()
                    }
                    _ => *prelude = BindingPrelude::default(),
                }
            } else {
                *prelude = BindingPrelude::default();
            }
        }
        let in_test = self.scopes().any(|sc| sc.kind == ScopeKind::Test);
        let can_run = match self.mode {
            RunMode::Normal => !in_test,
            RunMode::Test => in_test,
            RunMode::All => true,
        };
        let mut lines = VecDeque::from(flip_unsplit_lines(
            lines.into_iter().flat_map(split_words).collect(),
        ));
        while let Some(line) = lines.pop_front() {
            let assert_later = || {
                once(&line).chain(&lines).any(|line| {
                    line.iter()
                        .find(|w| w.value.is_code())
                        .is_some_and(|w| matches!(w.value, Word::Primitive(Primitive::Assert)))
                })
            };
            if line.is_empty()
                || !(can_run || must_run || assert_later() || words_should_run_anyway(&line))
            {
                continue;
            }
            let span =
                (line.first().unwrap().span.clone()).merge(line.last().unwrap().span.clone());
            if count_placeholders(&line) > 0 {
                self.add_error(
                    span.clone(),
                    "Cannot use placeholder outside of an index macro",
                );
            }
            // Compile the words
            let mut line_node = self.line(line)?;
            match line_node.sig() {
                Ok(sig) => {
                    // Update scope stack height
                    if let Ok(height) = &mut self.scope.stack_height {
                        *height = (*height + sig.outputs).saturating_sub(sig.args);
                        // Compile test assert
                        if self.mode != RunMode::Normal
                            && sig.outputs == 0
                            && !self
                                .scopes()
                                .any(|sc| sc.kind == ScopeKind::File(FileScopeKind::Git))
                        {
                            if let Some(Node::Prim(Primitive::Assert, span)) = line_node.last() {
                                let span = *span;
                                line_node.pop();
                                line_node.push(Node::ImplPrim(ImplPrimitive::TestAssert, span));
                            }
                        }
                    }
                }
                Err(e) => self.scope.stack_height = Err(span.sp(e)),
            }
            self.asm.root.push(line_node)
        }
        Ok(())
    }
    fn compile_bind_function(
        &mut self,
        name: Ident,
        local: LocalName,
        function: Function,
        span: usize,
        comment: Option<&str>,
    ) -> UiuaResult {
        let comment = comment.map(|text| {
            let comment = DocComment::from(text);
            if let Some(sig) = &comment.sig {
                if !sig.matches_sig(function.sig) {
                    self.emit_diagnostic(
                        format!(
                            "{}'s comment describes {}, \
                            but its code has signature {}",
                            name,
                            sig.sig_string(),
                            function.sig,
                        ),
                        DiagnosticKind::Warning,
                        self.get_span(span).clone().code().unwrap(),
                    );
                }
            }
            comment
        });
        self.scope.names.insert(name, local);
        self.asm.bind_function(local, function, span, comment);
        Ok(())
    }
    fn compile_bind_const(
        &mut self,
        name: Ident,
        local: LocalName,
        value: Option<Value>,
        span: usize,
        comment: Option<&str>,
    ) {
        let span = self.get_span(span).clone().code();
        let comment = comment.map(|text| {
            let comment = DocComment::from(text);
            if let Some(sig) = &comment.sig {
                self.emit_diagnostic(
                    format!(
                        "{}'s comment describes {}, \
                        but it is a constant",
                        name,
                        sig.sig_string(),
                    ),
                    DiagnosticKind::Warning,
                    span.clone().unwrap(),
                );
            }
            comment
        });
        self.asm
            .add_binding_at(local, BindingKind::Const(value), span, comment);
        self.scope.names.insert(name, local);
    }
    /// Import a module
    pub(crate) fn import_module(&mut self, path_str: &str, span: &CodeSpan) -> UiuaResult<PathBuf> {
        // Resolve path
        let (path, file_kind) = if let Some(mut url) = path_str.trim().strip_prefix("git:") {
            if url.contains("branch:") && url.contains("commit:") {
                return Err(self.fatal_error(
                    span.clone(),
                    "Cannot specify both branch and commit in git import",
                ));
            }
            let target = if let Some((a, b)) = url.split_once("branch:") {
                url = a;
                GitTarget::Branch(b.trim().into())
            } else if let Some((a, b)) = url.split_once("commit:") {
                url = a;
                GitTarget::Commit(b.trim().into())
            } else {
                GitTarget::Default
            };
            // Git import
            let mut url = url.trim().trim_end_matches(".git").to_string();
            if ![".com", ".net", ".org", ".io", ".dev"]
                .iter()
                .any(|s| url.contains(s))
            {
                if !url.starts_with('/') {
                    url = format!("/{url}");
                }
                url = format!("github.com{url}");
                self.emit_diagnostic(
                    "Implicit GitHub URLs are deprecated and \
                    will be removed in the future. Prefix the \
                    URL with `github.com/` or `https://github.com/`.",
                    DiagnosticKind::Warning,
                    span.clone(),
                );
            }
            if !(url.starts_with("https://") || url.starts_with("http://")) {
                url = format!("https://{url}");
            }
            self.code_meta
                .import_srcs
                .insert(span.clone(), ImportSrc::Git(url.clone()));
            let path = self
                .backend()
                .load_git_module(&url, target)
                .map_err(|e| self.fatal_error(span.clone(), e))?;
            (path, FileScopeKind::Git)
        } else {
            // Normal import
            let path = self.resolve_import_path(Path::new(path_str));
            self.code_meta
                .import_srcs
                .insert(span.clone(), ImportSrc::File(path.clone()));
            (path, FileScopeKind::Source)
        };
        if !self.imports.contains_key(&path) {
            // We cache Git modules on WASM so that the pad doesn't have to recompile big modules constantly
            thread_local! {
                static GIT_CACHE: RefCell<HashMap<PathBuf, Compiler>> = RefCell::new(HashMap::new());
            }
            let bytes = self
                .backend()
                .file_read_all(&path)
                .or_else(|e| {
                    if path.ends_with(Path::new("example.ua")) {
                        Ok(EXAMPLE_UA.as_bytes().to_vec())
                    } else {
                        Err(e)
                    }
                })
                .map_err(|e| self.fatal_error(span.clone(), e))?;
            if let Some(mut comp) = (bytes.len() > 1000)
                .then(|| GIT_CACHE.with(|cache| cache.borrow().get(&path).cloned()))
                .flatten()
            {
                swap(self, &mut comp);
                self.macro_env.rt.backend = comp.macro_env.rt.backend;
                self.asm.inputs.strings = comp.asm.inputs.strings;
                self.asm.inputs.files.extend(comp.asm.inputs.files);
                self.scope.experimental = comp.scope.experimental;
                self.diagnostics.extend(comp.diagnostics);
            } else {
                let input: EcoString = String::from_utf8(bytes)
                    .map_err(|e| {
                        self.fatal_error(span.clone(), format!("Failed to read file: {e}"))
                    })?
                    .into();
                if self.current_imports.iter().any(|p| p == &path) {
                    return Err(self.fatal_error(
                        span.clone(),
                        format!("Cycle detected importing {}", path.to_string_lossy()),
                    ));
                }
                let (module, ()) = self.in_scope(ScopeKind::File(file_kind), |comp| {
                    comp.load_str_src(&input, &path).map(drop)
                })?;
                self.imports.insert(path.clone(), module);
                #[cfg(target_arch = "wasm32")]
                if file_kind == FileScopeKind::Git {
                    GIT_CACHE.with(|cache| {
                        let mut clone = self.clone();
                        clone.macro_env.rt.backend = Arc::new(crate::SafeSys::default());
                        cache.borrow_mut().insert(path.clone(), clone);
                    });
                }
            };
        }
        let module = self.imports.get(&path).unwrap();
        if module.experimental {
            self.experimental_error(span, || {
                format!(
                    "Module `{path_str}` is experimental. \
                    To use it, add `# Experimental!` to the top of this file."
                )
            });
        }
        Ok(path)
    }
    /// Resolve a declared import path relative to the path of the file that is being executed
    pub(crate) fn resolve_import_path(&self, path: &Path) -> PathBuf {
        let mut target = if let Some(parent) = self.current_imports.last().and_then(|p| p.parent())
        {
            parent.join(path)
        } else {
            path.to_path_buf()
        };
        if !target.exists() && target.extension().is_none() {
            target = target.with_extension("ua");
        }
        let base = Path::new(".");
        if let (Ok(canon_target), Ok(canon_base)) = (target.canonicalize(), base.canonicalize()) {
            pathdiff::diff_paths(canon_target, canon_base).unwrap_or(target)
        } else {
            pathdiff::diff_paths(&target, base).unwrap_or(target)
        }
    }
    // Compile a line, checking an end-of-line signature comment
    fn line(&mut self, line: Vec<Sp<Word>>) -> UiuaResult<Node> {
        let comment_sig = line_sig(&line);
        let is_empty = line.iter().all(|w| !w.value.is_code());
        let node = self.words(line)?;
        if let Some(comment_sig) = comment_sig {
            if let Ok(sig) = node.sig() {
                if !is_empty && !comment_sig.value.matches_sig(sig) {
                    self.emit_diagnostic(
                        format!("Line signature {sig} does not match comment"),
                        DiagnosticKind::Warning,
                        comment_sig.span.clone(),
                    );
                }
            }
        }
        Ok(node)
    }
    fn args(&mut self, words: Vec<Sp<Word>>) -> UiuaResult<EcoVec<SigNode>> {
        words
            .into_iter()
            .filter(|w| w.value.is_code())
            .map(|w| self.word_sig(w))
            .collect()
    }
    fn words_sig(&mut self, words: Vec<Sp<Word>>) -> UiuaResult<SigNode> {
        let span = words
            .first()
            .zip(words.last())
            .map(|(f, l)| f.span.clone().merge(l.span.clone()));
        let node = self.words(words)?;
        let sig = if let Some(span) = span {
            self.sig_of(&node, &span)?
        } else {
            Signature::new(0, 0)
        };
        Ok(SigNode::new(node, sig))
    }
    fn words(&mut self, mut words: Vec<Sp<Word>>) -> UiuaResult<Node> {
        words.retain(|word| word.value.is_code());
        words.reverse();
        #[derive(Debug, Clone)]
        struct PrevWord(Option<Primitive>, Option<Signature>, CodeSpan);
        let mut a: Option<PrevWord> = None;
        let mut b: Option<PrevWord> = None;
        let mut nodes = Node::empty();
        for word in flip_unsplit_lines(split_words(words)).into_iter().flatten() {
            let span = word.span.clone();
            let prim = match word.value {
                Word::Primitive(prim) => Some(prim),
                _ => None,
            };

            match (a, &b, prim) {
                // Flip monadic dup diagnostic
                (
                    Some(PrevWord(Some(Primitive::Dup), _, a_span)),
                    Some(PrevWord(
                        _,
                        Some(Signature {
                            args: 1,
                            outputs: 1,
                        }),
                        _,
                    )),
                    Some(Primitive::Flip),
                ) => {
                    self.emit_diagnostic(
                        format!(
                            "Prefer {} over {} {} here",
                            Primitive::On,
                            Primitive::Flip,
                            Primitive::Dup
                        ),
                        DiagnosticKind::Style,
                        a_span.merge(span.clone()),
                    );
                }
                // Keep unique dup diagnostic
                (
                    Some(PrevWord(Some(Primitive::Dup), _, a_span)),
                    Some(PrevWord(Some(Primitive::Unique), _, _)),
                    Some(Primitive::Keep),
                ) => {
                    self.emit_diagnostic(
                        format!(
                            "Prefer {} over {}{}{}",
                            Primitive::Deduplicate.format(),
                            Primitive::Keep,
                            Primitive::Unique,
                            Primitive::Dup
                        ),
                        DiagnosticKind::Advice,
                        a_span.merge(span.clone()),
                    );
                }
                _ => {}
            }

            // Compile the word
            let node = self.word(word)?;
            let sig = node.sig().ok();
            nodes.push(node);
            a = b;
            b = Some(PrevWord(prim, sig, span));
        }
        Ok(nodes)
    }
    fn word_sig(&mut self, word: Sp<Word>) -> UiuaResult<SigNode> {
        let span = word.span.clone();
        let node = self.word(word)?;
        let sig = self.sig_of(&node, &span)?;
        Ok(SigNode::new(node, sig))
    }
    fn word(&mut self, word: Sp<Word>) -> UiuaResult<Node> {
        Ok(match word.value {
            Word::Number(_, n) => Node::new_push(n),
            Word::Char(c) => {
                let val: Value = if c.chars().count() == 1 {
                    c.chars().next().unwrap().into()
                } else {
                    c.into()
                };
                Node::new_push(val)
            }
            Word::String(s) => Node::new_push(s),
            Word::MultilineString(lines) => {
                let mut s = EcoVec::new();
                for (i, line) in lines.into_iter().enumerate() {
                    if i > 0 {
                        s.push('\n');
                    }
                    s.extend(line.value.chars());
                }
                Node::new_push(s)
            }
            Word::Label(label) => Node::Label(label.into(), self.add_span(word.span.clone())),
            Word::FormatString(frags) => {
                let parts = frags.into_iter().map(Into::into).collect();
                let span = self.add_span(word.span.clone());
                Node::Format(parts, span)
            }
            Word::MultilineFormatString(lines) => {
                let span = self.add_span(word.span.clone());
                let mut curr_part = EcoString::new();
                let mut parts = EcoVec::new();
                for (l, line) in lines.into_iter().enumerate() {
                    if l > 0 {
                        curr_part.push('\n');
                    }
                    for (f, frag) in line.value.into_iter().enumerate() {
                        if f > 0 {
                            parts.push(take(&mut curr_part));
                        }
                        curr_part.push_str(&frag);
                    }
                }
                parts.push(curr_part);
                Node::Format(parts, span)
            }
            Word::Ref(r) => self.reference(r)?,
            Word::IncompleteRef { path, in_macro_arg } => {
                if let Some((_, locals)) = self.ref_path(&path, in_macro_arg)? {
                    self.add_error(
                        path.last().unwrap().tilde_span.clone(),
                        "Incomplete module reference",
                    );
                    for (local, comp) in locals.iter().zip(path) {
                        self.validate_local(&comp.module.value, *local, &comp.module.span);
                        self.code_meta
                            .global_references
                            .insert(comp.module.span, local.index);
                    }
                    self.code_meta
                        .incomplete_refs
                        .insert(word.span.clone(), locals.last().unwrap().index);
                }
                Node::empty()
            }
            Word::Strand(items) => {
                // Track span for LSP
                let just_spans: Vec<_> = items.iter().map(|w| w.span.clone()).collect();
                // Compile individual items
                let op_nodes = items
                    .into_iter()
                    .rev()
                    .map(|word| self.word_sig(word))
                    .collect::<UiuaResult<Vec<_>>>()?;
                // Check item sigs
                let has_functions = op_nodes.iter().any(|sn| sn.sig.args > 0);
                if has_functions {
                    return Err(
                        self.fatal_error(word.span.clone(), "Functions are not allowed in strands")
                    );
                }
                self.code_meta.strands.insert(word.span.clone(), just_spans);
                // Flatten instrs
                let inner = Node::from_iter(op_nodes.into_iter().map(|sn| sn.node));

                // Normal strand

                // Diagnostic for strand of characters
                if !inner.is_empty()
                    && inner.iter().all(
                        |instr| matches!(instr, Node::Push(Value::Char(arr)) if arr.rank() == 0),
                    )
                {
                    self.emit_diagnostic(
                        "Stranded characters should instead be written as a string",
                        DiagnosticKind::Advice,
                        word.span.clone(),
                    );
                }

                let span = self.add_span(word.span.clone());
                // Inline constant arrays
                if inner.iter().all(|instr| matches!(instr, Node::Push(_))) {
                    let values: Vec<_> = inner
                        .iter()
                        .rev()
                        .map(|instr| match instr {
                            Node::Push(v) => v.clone(),
                            _ => unreachable!(),
                        })
                        .collect();
                    match Value::from_row_values(values, &(&word.span, &self.asm.inputs)) {
                        Ok(val) => return Ok(Node::new_push(val)),
                        Err(e) if e.is_fill => {}
                        Err(e) => return Err(e),
                    }
                }
                let sig = self.sig_of(&inner, &word.span)?;
                Node::Array {
                    len: sig.outputs,
                    inner: inner.into(),
                    boxed: false,
                    span,
                }
            }
            Word::Array(arr) => {
                // Track span for LSP
                if !arr.boxes
                    && (arr.lines.iter().flatten())
                        .filter(|w| w.value.is_code())
                        .all(|w| w.value.is_literal() && !matches!(w.value, Word::Strand(_)))
                {
                    let just_spans: Vec<_> = (arr.lines.iter().rev().flatten())
                        .filter(|w| w.value.is_code())
                        .map(|w| w.span.clone())
                        .collect();
                    self.code_meta
                        .array_inner_spans
                        .insert(word.span.clone(), just_spans);
                }
                let line_count = arr.lines.len();
                let mut inner = Node::empty();
                for line in arr.lines.into_iter().rev() {
                    inner.push(self.line(line)?);
                }
                // Validate inner loop correctness
                let inner_sig = self.validate_array_loop_sig(&inner, &word.span);
                // Validate signature
                if let Some((declared_sig, inner_sig)) = arr.signature.zip(inner_sig) {
                    if inner_sig != declared_sig.value {
                        self.add_error(
                            declared_sig.span.clone(),
                            format!(
                                "Array signature mismatch: declared {} but inferred {}",
                                declared_sig.value, inner_sig
                            ),
                        );
                    }
                }
                // Diagnostic for array of characters
                if line_count <= 1
                    && !arr.boxes
                    && !inner.is_empty()
                    && inner.iter().all(
                        |instr| matches!(instr, Node::Push(Value::Char(arr)) if arr.rank() == 0),
                    )
                {
                    self.emit_diagnostic(
                        "An array of characters should instead be written as a string",
                        DiagnosticKind::Advice,
                        word.span.clone(),
                    );
                }
                let span = self.add_span(word.span.clone());
                // Inline constant arrays
                if inner.iter().all(|instr| matches!(instr, Node::Push(_))) {
                    let empty = inner.is_empty();
                    let values = inner.iter().rev().map(|instr| match instr {
                        Node::Push(v) => v.clone(),
                        _ => unreachable!(),
                    });
                    let res = if arr.boxes {
                        if empty {
                            Ok(Array::<Boxed>::default().into())
                        } else {
                            Value::from_row_values(
                                values
                                    .map(|v| Value::Box(Boxed(v).into()))
                                    .collect::<Vec<_>>(),
                                &(&word.span, &self.asm.inputs),
                            )
                        }
                    } else {
                        Value::from_row_values(
                            values.collect::<Vec<_>>(),
                            &(&word.span, &self.asm.inputs),
                        )
                    };
                    match res {
                        Ok(val) => {
                            self.code_meta
                                .array_shapes
                                .insert(word.span.clone(), val.shape().clone());
                            return Ok(Node::new_push(val));
                        }
                        Err(e) if e.is_fill => {}
                        Err(e) => return Err(e),
                    }
                }
                // Normal case
                let sig = self.sig_of(&inner, &word.span)?;
                Node::Array {
                    len: sig.outputs,
                    inner: inner.into(),
                    boxed: arr.boxes,
                    span,
                }
            }
            Word::Func(func) => self.func(func, word.span)?,
            Word::Pack(pack) => {
                self.add_error(
                    word.span.clone(),
                    "Function packs are not allowed without a modifier",
                );
                if let Some(first) = pack.branches.into_iter().next() {
                    self.word(first.map(Word::Func))?
                } else {
                    Node::empty()
                }
            }
            Word::Primitive(p) => self.primitive(p, word.span),
            Word::Modified(m) => self.modified(*m, None)?,
            Word::Placeholder(_) => {
                // We could error here, but it's easier to handle it higher up
                Node::empty()
            }
            Word::SemanticComment(sc) => {
                self.semantic_comment(sc, word.span);
                Node::empty()
            }
            Word::OutputComment { i, n } => Node::SetOutputComment { i, n },
            Word::Subscript(sub) => self.subscript(*sub, word.span)?,
            Word::Comment(_) | Word::Spaces | Word::BreakLine | Word::FlipLine => Node::empty(),
        })
    }
    fn semantic_comment(&mut self, comment: SemanticComment, span: CodeSpan) {
        match comment {
            SemanticComment::Experimental => {
                self.scope.experimental = true;
            }
            SemanticComment::NoInline => {
                todo!("# No inline!")
            }
            SemanticComment::TrackCaller => {
                todo!("# Track caller!")
            }
            SemanticComment::Boo => {
                self.add_error(span, "The compiler is scared!");
            }
        }
    }
    /// Emit a warning if a loop inside an array could
    /// potentially pull in a variable number of values
    fn validate_array_loop_sig(&mut self, node: &Node, span: &CodeSpan) -> Option<Signature> {
        let inner_sig = node.sig();
        if self.current_bindings.is_empty() && !matches!(self.scope.kind, ScopeKind::Temp(_)) {
            return inner_sig.ok();
        }
        let Err(e) = &inner_sig else {
            // Case where repeat's function has a balanced signature
            // This is fine in other contexts, so an error is not returned
            // from the signature check, but it is not okay in an array.
            if let Some(i) = node
                .iter()
                .position(|instr| {
                    matches!(
                        instr,
                        Node::Mod(Primitive::Repeat, ..)
                            | Node::ImplMod(ImplPrimitive::RepeatWithInverse, ..)
                    )
                })
                .filter(|&i| i > 0)
            {
                let (Node::Mod(_, args, _) | Node::ImplMod(_, args, _)) = &node[i - 1] else {
                    unreachable!()
                };
                let body_sig = args[0].sig;
                let before_sig = nodes_sig(&node[..i - 1]).ok()?;
                let after_sig = nodes_sig(&node[i + 1..]).ok()?;
                if body_sig.args == body_sig.outputs
                    && before_sig.args < body_sig.args
                    && before_sig.outputs <= body_sig.args
                    && after_sig.args.saturating_sub(body_sig.outputs) < body_sig.args
                {
                    let replacement: String = repeat('⊙')
                        .take(body_sig.args.saturating_sub(1))
                        .chain(['∘'])
                        .collect();
                    let message = format!(
                        "This array contains a loop with an equal number \
                        of arguments and outputs. This may result in a \
                        variable number of values being pulled into the \
                        array. To fix this, insert `{replacement}` on the \
                        right side of the array.",
                    );
                    self.emit_diagnostic(message, DiagnosticKind::Warning, span.clone());
                }
            }
            return inner_sig.ok();
        };
        let before_sig = (0..node.len())
            .rev()
            .find_map(|i| nodes_sig(&node[..i]).ok())
            .unwrap();
        let after_sig = (0..=node.len())
            .find_map(|i| nodes_sig(&node[i..]).ok())
            .unwrap();
        match e.kind {
            SigCheckErrorKind::LoopVariable { sig: body_sig, inf } => {
                let positive_body_sig = body_sig.outputs >= body_sig.args;
                let balanced_after_args = body_sig.args.saturating_sub(before_sig.outputs);
                if body_sig.args > 0
                    && (positive_body_sig && after_sig.args != balanced_after_args && !inf)
                    || body_sig.args == 0 && after_sig.args > 0
                {
                    let mut message = format!(
                        "This array contains a loop that has a variable \
                        number of outputs. The code left of the loop has \
                        signature {after_sig}, which may result in a variable \
                        number of values being pulled into the array."
                    );
                    let max_after_args_required_by_before = before_sig
                        .outputs
                        .saturating_sub(before_sig.args)
                        .max(before_sig.args);
                    if after_sig.args > max_after_args_required_by_before {
                        if after_sig.args > body_sig.args {
                            let replacement: String = repeat('⊙')
                                .take(after_sig.args - body_sig.args)
                                .chain(['∘'])
                                .collect();
                            message.push_str(&format!(
                                " To fix this, insert `{replacement}` on the \
                                right side of the array."
                            ));
                        } else {
                            let replacement: String =
                                repeat('⊙').take(body_sig.args - 1).chain(['∘']).collect();
                            message.push_str(&format!(
                                " To fix this, insert `{replacement}` to the \
                                left of the loop."
                            ));
                        }
                        self.emit_diagnostic(message, DiagnosticKind::Warning, span.clone())
                    }
                }
            }
            SigCheckErrorKind::LoopOverreach => self.emit_diagnostic(
                "This array contains a loop that has a variable \
                number of inputs. This may result in a variable \
                number of values being pulled into the array.",
                DiagnosticKind::Warning,
                span.clone(),
            ),
            _ => {}
        }
        inner_sig.ok()
    }
    /// Find the [`LocalName`]s of both the name and all parts of the path of a [`Ref`]
    ///
    /// Returns [`None`] if the reference is to a constant
    fn ref_local(&self, r: &Ref) -> UiuaResult<Option<(Vec<LocalName>, LocalName)>> {
        if let Some((module, path_locals)) = self.ref_path(&r.path, r.in_macro_arg)? {
            if let Some(local) = module.names.get(&r.name.value).copied().or_else(|| {
                (r.name.value.strip_suffix('!')).and_then(|name| {
                    module.names.get(name).copied().filter(|local| {
                        matches!(&self.asm.bindings[local.index].kind, BindingKind::Module(_))
                    })
                })
            }) {
                Ok(Some((path_locals, local)))
            } else {
                Err(self.fatal_error(
                    r.name.span.clone(),
                    format!("Item `{}` not found", r.name.value,),
                ))
            }
        } else if let Some(local) = self.find_name(&r.name.value, r.in_macro_arg) {
            Ok(Some((Vec::new(), local)))
        } else if r.path.is_empty() && CONSTANTS.iter().any(|def| def.name == r.name.value) {
            Ok(None)
        } else {
            Err(self.fatal_error(
                r.name.span.clone(),
                format!("Unknown identifier `{}`", r.name.value),
            ))
        }
    }
    fn find_name(&self, name: &str, skip_local: bool) -> Option<LocalName> {
        if !skip_local {
            if let Some(local) = self.scope.names.get(name).copied() {
                return Some(local);
            }
        }
        let mut hit_file = false;
        for scope in self.higher_scopes.iter().rev() {
            if matches!(scope.kind, ScopeKind::File(_)) {
                if hit_file {
                    break;
                }
                hit_file = true;
            }
            if let Some(local) = scope.names.get(name).copied() {
                return Some(local);
            }
        }
        // Attempt to look up the identifier as a non-macro
        let as_non_macro = self.find_name(name.strip_suffix('!')?, skip_local)?;
        if let BindingKind::Module(_) | BindingKind::Import(_) =
            self.asm.bindings[as_non_macro.index].kind
        {
            // Only allow it if it is a module
            Some(as_non_macro)
        } else {
            None
        }
    }
    fn ref_path(
        &self,
        path: &[RefComponent],
        skip_local: bool,
    ) -> UiuaResult<Option<(&Module, Vec<LocalName>)>> {
        let Some(first) = path.first() else {
            return Ok(None);
        };
        let mut path_locals = Vec::new();
        let module_local = self
            .find_name(&first.module.value, skip_local)
            .ok_or_else(|| {
                self.fatal_error(
                    first.module.span.clone(),
                    format!("Unknown import `{}`", first.module.value),
                )
            })?;
        path_locals.push(module_local);
        let global = &self.asm.bindings[module_local.index].kind;
        let mut module = match global {
            BindingKind::Import(path) => &self.imports[path],
            BindingKind::Module(module) => module,
            BindingKind::Func(_) => {
                return Err(self.fatal_error(
                    first.module.span.clone(),
                    format!("`{}` is a function, not a module", first.module.value),
                ))
            }
            BindingKind::Const(_) => {
                return Err(self.fatal_error(
                    first.module.span.clone(),
                    format!("`{}` is a constant, not a module", first.module.value),
                ))
            }
            BindingKind::IndexMacro(_) => {
                return Err(self.fatal_error(
                    first.module.span.clone(),
                    format!("`{}` is an index macro, not a module", first.module.value),
                ))
            }
            BindingKind::CodeMacro(_) => {
                return Err(self.fatal_error(
                    first.module.span.clone(),
                    format!("`{}` is a code macro, not a module", first.module.value),
                ))
            }
        };
        for comp in path.iter().skip(1) {
            let submod_local = module
                .names
                .get(&comp.module.value)
                .copied()
                .ok_or_else(|| {
                    self.fatal_error(
                        comp.module.span.clone(),
                        format!("Module `{}` not found", comp.module.value),
                    )
                })?;
            path_locals.push(submod_local);
            let global = &self.asm.bindings[submod_local.index].kind;
            module = match global {
                BindingKind::Import(path) => &self.imports[path],
                BindingKind::Module(module) => module,
                BindingKind::Func(_) => {
                    return Err(self.fatal_error(
                        comp.module.span.clone(),
                        format!("`{}` is a function, not a module", comp.module.value),
                    ))
                }
                BindingKind::Const(_) => {
                    return Err(self.fatal_error(
                        comp.module.span.clone(),
                        format!("`{}` is a constant, not a module", comp.module.value),
                    ))
                }
                BindingKind::IndexMacro(_) => {
                    return Err(self.fatal_error(
                        comp.module.span.clone(),
                        format!(
                            "`{}` is a positional macro, not a module",
                            comp.module.value
                        ),
                    ))
                }
                BindingKind::CodeMacro(_) => {
                    return Err(self.fatal_error(
                        comp.module.span.clone(),
                        format!("`{}` is a code macro, not a module", comp.module.value),
                    ))
                }
            };
        }

        Ok(Some((module, path_locals)))
    }
    fn reference(&mut self, r: Ref) -> UiuaResult<Node> {
        if r.path.is_empty() {
            self.ident(r.name.value, r.name.span, r.in_macro_arg)
        } else if let Some((path_locals, local)) = self.ref_local(&r)? {
            self.validate_local(&r.name.value, local, &r.name.span);
            for (local, comp) in path_locals.into_iter().zip(&r.path) {
                self.validate_local(&comp.module.value, local, &comp.module.span);
                (self.code_meta.global_references).insert(comp.module.span.clone(), local.index);
            }
            self.code_meta
                .global_references
                .insert(r.name.span.clone(), local.index);
            Ok(self.global_index(local.index, r.name.span))
        } else {
            self.ident(r.name.value, r.name.span, r.in_macro_arg)
        }
    }
    fn ident(&mut self, ident: Ident, span: CodeSpan, skip_local: bool) -> UiuaResult<Node> {
        Ok(
            if let Some(curr) = (self.current_bindings.last_mut()).filter(|curr| curr.name == ident)
            {
                // Name is a recursive call
                let Some(sig) = curr.signature else {
                    return Err(self.fatal_error(
                        span,
                        format!(
                            "Recursive function `{ident}` must have a \
                        signature declared after the ←."
                        ),
                    ));
                };
                curr.referenced = true;
                (self.code_meta.global_references).insert(span.clone(), curr.global_index);
                Node::CallGlobal(curr.global_index, sig)
            } else if let Some(local) = self.find_name(&ident, skip_local) {
                // Name exists in scope
                self.validate_local(&ident, local, &span);
                (self.code_meta.global_references).insert(span.clone(), local.index);
                self.global_index(local.index, span)
            } else if let Some(constant) = CONSTANTS.iter().find(|c| c.name == ident) {
                // Name is a built-in constant
                self.code_meta
                    .constant_references
                    .insert(span.clone().sp(ident));
                Node::Push(
                    constant
                        .value
                        .resolve(self.scope_file_path(), &*self.backend()),
                )
            } else {
                return Err(self.fatal_error(span, format!("Unknown identifier `{ident}`")));
            },
        )
    }
    fn scope_file_path(&self) -> Option<&Path> {
        for scope in self.scopes() {
            if let Some(file_path) = &scope.file_path {
                return Some(file_path);
            }
        }
        None
    }
    fn global_index(&mut self, index: usize, span: CodeSpan) -> Node {
        let global = self.asm.bindings[index].kind.clone();
        match global {
            BindingKind::Const(Some(val)) => Node::new_push(val),
            BindingKind::Const(None) => Node::CallGlobal(index, Signature::new(0, 1)),
            BindingKind::Func(f) => Node::Call(f, self.add_span(span)),
            BindingKind::Import(path) => {
                if let Some(local) = self.imports.get(&path).and_then(|m| m.names.get("Call")) {
                    self.code_meta.global_references.remove(&span);
                    self.code_meta
                        .global_references
                        .insert(span.clone(), local.index);
                    self.global_index(local.index, span)
                } else {
                    self.add_error(
                        span,
                        "Module cannot be called here as \
                        it has no `Call` function.",
                    );
                    Node::empty()
                }
            }
            BindingKind::Module(m) => {
                if let Some(local) = m.names.get("Call").or_else(|| m.names.get("New")) {
                    self.code_meta.global_references.remove(&span);
                    self.code_meta
                        .global_references
                        .insert(span.clone(), local.index);
                    self.global_index(local.index, span)
                } else {
                    self.add_error(
                        span,
                        "Module cannot be called here as \
                        it has no `Call` or `New` function.",
                    );
                    Node::empty()
                }
            }
            BindingKind::IndexMacro(_) | BindingKind::CodeMacro(_) => {
                // We could error here, but it's easier to handle it higher up
                Node::empty()
            }
        }
    }
    fn func(&mut self, func: Func, span: CodeSpan) -> UiuaResult<Node> {
        if (func.lines.iter().flatten().filter(|w| w.value.is_code())).count() == 1 {
            // Inline single ident
            if let Some(
                word @ Sp {
                    value: Word::Ref(_),
                    ..
                },
            ) = func.lines.iter().flatten().find(|w| w.value.is_code())
            {
                return self.word(word.clone());
            }
        }

        let mut root = Node::empty();
        for line in func.lines {
            root.push(self.line(line)?);
        }

        // Validate signature
        let sig = match root.sig() {
            Ok(mut sig) => {
                if let Some(declared_sig) = &func.signature {
                    if declared_sig.value == sig {
                        sig = declared_sig.value;
                    } else {
                        self.add_error(
                            declared_sig.span.clone(),
                            format!(
                                "Function signature mismatch: declared {} but inferred {}",
                                declared_sig.value, sig
                            ),
                        );
                    }
                }
                Some(sig)
            }
            Err(e) => {
                if let Some(declared_sig) = &func.signature {
                    if e.kind == SigCheckErrorKind::Ambiguous {
                        Some(declared_sig.value)
                    } else {
                        return Err(self.fatal_error(
                            declared_sig.span.clone(),
                            format!(
                                "Cannot infer function signature: {e}. \
                                An explicit signature can only be used \
                                with ambiguous functions."
                            ),
                        ));
                    }
                } else {
                    None
                }
            }
        };
        if let Some(sig) = sig {
            self.code_meta.function_sigs.insert(
                span.clone(),
                SigDecl {
                    sig,
                    explicit: func.signature.is_some(),
                    inline: true,
                    set_inverses: Default::default(),
                },
            );
        }
        Ok(root)
    }
    fn switch(&mut self, branches: Vec<Sp<Word>>, span: CodeSpan) -> UiuaResult<Node> {
        let count = branches.len();
        // Compile branches
        let mut br = EcoVec::with_capacity(count);
        let mut rigid_indices = Vec::new();
        let mut flex_indices = Vec::new();
        for (i, branch) in branches.into_iter().enumerate() {
            let span = branch.span.clone();
            let SigNode { node, sig } = self.word_sig(branch)?;
            let is_flex = node
                .iter()
                .rposition(|node| matches!(node, Node::Prim(Primitive::Assert, _)))
                .is_some_and(|end| {
                    (0..end).rev().any(|start| {
                        let sub = node.slice(start..end);
                        match sub.as_slice() {
                            [Node::Push(val), Node::Prim(Primitive::Dup, _)]
                            | [Node::Push(val), Node::Push(..)]
                                if val != &Value::from(1) =>
                            {
                                return true;
                            }
                            [Node::Format(..), Node::Prim(Primitive::Dup, _)] => return true,
                            _ => (),
                        }
                        if !(sub.is_pure(Purity::Pure, &self.asm)
                            && sub.sig().is_ok_and(|sig| sig == (0, 2)))
                        {
                            return false;
                        }
                        todo!("switch flex eval")
                    })
                });
            br.push((SigNode::new(node, sig), span));
            if is_flex {
                flex_indices.push(i);
            } else {
                rigid_indices.push(i);
            }
        }
        let mut rigid_funcs = rigid_indices.into_iter().map(|i| &br[i]);
        let mut sig = None;
        if let Some((arg, _)) = rigid_funcs.next() {
            sig = Some(arg.sig);
            let sig = sig.as_mut().unwrap();
            // Compile remaining branches
            for (arg, span) in rigid_funcs {
                if arg.sig.is_compatible_with(*sig) {
                    *sig = sig.max_with(arg.sig);
                } else if arg.sig.outputs == sig.outputs {
                    sig.args = sig.args.max(arg.sig.args)
                } else {
                    self.add_error(
                        span.clone(),
                        format!(
                            "Switch branch's signature {} is \
                            incompatible with previous branches {sig}",
                            arg.sig
                        ),
                    );
                }
            }
        }
        let mut flex_funcs = flex_indices.into_iter().map(|i| &br[i]);
        let mut sig = sig.unwrap_or_else(|| flex_funcs.next().unwrap().0.sig);
        for (arg, _) in flex_funcs {
            sig.args = sig.args.max(arg.sig.args);
        }

        let span = self.add_span(span.clone());
        Ok(Node::Switch {
            branches: br.into_iter().map(|(arg, _)| arg).collect(),
            sig,
            span,
            under_cond: false,
        })
    }
    fn handle_primitive_deprecation(&mut self, prim: Primitive, span: &CodeSpan) {
        if let Some(suggestion) = prim.deprecation_suggestion() {
            if !self.deprecated_prim_errors.insert(prim) {
                return;
            }
            let suggestion = if suggestion.is_empty() {
                String::new()
            } else {
                format!(", {suggestion}")
            };
            self.emit_diagnostic(
                format!(
                    "{} is deprecated and will be removed in a future version{}",
                    prim.format(),
                    suggestion
                ),
                DiagnosticKind::Warning,
                span.clone(),
            );
        }
    }
    fn handle_primitive_experimental(&mut self, prim: Primitive, span: &CodeSpan) {
        if prim.is_experimental() {
            self.experimental_error(span, || {
                format!(
                    "{} is experimental. To use it, add \
                    `# Experimental!` to the top of the file.",
                    prim.format()
                )
            });
        }
    }
    fn validate_primitive(&mut self, prim: Primitive, span: &CodeSpan) {
        self.handle_primitive_experimental(prim, span);
        self.handle_primitive_deprecation(prim, span);
    }
    fn primitive(&mut self, prim: Primitive, span: CodeSpan) -> Node {
        self.validate_primitive(prim, &span);
        let span = self.add_span(span);
        Node::Prim(prim, span)
    }
    #[allow(clippy::match_single_binding)]
    fn subscript(&mut self, sub: Subscript, span: CodeSpan) -> UiuaResult<Node> {
        if !matches!(sub.word.value, Word::Primitive(Primitive::Utf8)) {
            self.experimental_error(&span, || {
                "Subscripts are experimental. To use them, add \
                `# Experimental!` to the top of the file."
            });
        }
        let Some(n) = sub.n.value else {
            return self.word(sub.word);
        };
        Ok(match sub.word.value {
            Word::Modified(m) => match m.modifier.value {
                Modifier::Ref(_) => {
                    self.add_error(span, "Subscripts are not implemented for macros");
                    self.modified(*m, Some(n))?
                }
                Modifier::Primitive(prim) => match prim {
                    _ => {
                        if !matches!(
                            prim,
                            Primitive::Both
                                | Primitive::Repeat
                                | Primitive::Rows
                                | Primitive::Each
                                | Primitive::Tuples
                        ) {
                            self.add_error(
                                span,
                                format!("Subscripts are not implemented for {}", prim.format()),
                            );
                        }
                        self.modified(*m, Some(n))?
                    }
                },
            },
            Word::Primitive(prim) => match prim {
                prim if prim.sig().is_some_and(|sig| sig == (2, 1))
                    && prim.subscript_sig(Some(2)).is_some_and(|sig| sig == (1, 1)) =>
                {
                    Node::from_iter([
                        self.word(sub.n.span.sp(Word::Number(n.to_string(), n as f64)))?,
                        self.primitive(prim, span),
                    ])
                }
                Primitive::Transpose => {
                    if n > 100 {
                        self.add_error(span.clone(), "Too many subscript repetitions");
                    }
                    (0..n.min(100))
                        .map(|_| self.primitive(prim, span.clone()))
                        .collect()
                }
                Primitive::Sqrt => {
                    if n == 0 {
                        self.add_error(span.clone(), "Cannot take 0th root");
                    }
                    Node::from_iter([
                        Node::new_push(1.0 / n.max(1) as f64),
                        self.primitive(Primitive::Pow, span),
                    ])
                }
                Primitive::Round | Primitive::Floor | Primitive::Ceil => {
                    let mul = 10f64.powi(n as i32);
                    Node::from_iter([
                        Node::new_push(mul),
                        self.primitive(Primitive::Mul, span.clone()),
                        self.primitive(prim, span.clone()),
                        Node::new_push(mul),
                        self.primitive(Primitive::Div, span),
                    ])
                }
                Primitive::Rand => Node::from_iter([
                    self.primitive(Primitive::Rand, span.clone()),
                    Node::new_push(n),
                    self.primitive(Primitive::Mul, span.clone()),
                    self.primitive(Primitive::Floor, span),
                ]),
                Primitive::Utf8 => {
                    if n != 8 {
                        self.add_error(span.clone(), "Only UTF-8 is supported");
                    }
                    self.primitive(prim, span)
                }
                Primitive::Couple => match n {
                    1 => self.primitive(Primitive::Fix, span),
                    2 => self.primitive(Primitive::Couple, span),
                    n => Node::Array {
                        len: n,
                        inner: Node::empty().into(),
                        boxed: false,
                        span: self.add_span(span.clone()),
                    },
                },
                Primitive::Box => Node::Array {
                    len: n,
                    inner: Node::empty().into(),
                    boxed: true,
                    span: self.add_span(span.clone()),
                },
                Primitive::Stack => Node::ImplPrim(
                    ImplPrimitive::TraceN {
                        n,
                        inverse: false,
                        stack_sub: true,
                    },
                    self.add_span(span.clone()),
                ),
                _ => {
                    self.add_error(
                        span.clone(),
                        format!("Subscripts are not implemented for {}", prim.format()),
                    );
                    self.primitive(prim, span)
                }
            },
            _ => {
                self.add_error(span.clone(), "Subscripts are not allowed in this context");
                self.word(sub.word)?
            }
        })
    }
    /// Get all diagnostics
    pub fn diagnostics(&self) -> &BTreeSet<Diagnostic> {
        &self.diagnostics
    }
    /// Get all diagnostics mutably
    pub fn diagnostics_mut(&mut self) -> &mut BTreeSet<Diagnostic> {
        &mut self.diagnostics
    }
    /// Take all diagnostics
    ///
    /// These are only available if `print_diagnostics` is `false`
    pub fn take_diagnostics(&mut self) -> BTreeSet<Diagnostic> {
        take(&mut self.diagnostics)
    }
    /// Construct and add a diagnostic with a custom span
    pub fn emit_diagnostic(
        &mut self,
        message: impl Into<String>,
        kind: DiagnosticKind,
        span: impl Into<Span>,
    ) {
        let inputs = self.asm.inputs.clone();
        self.emit_diagnostic_impl(Diagnostic::new(message.into(), span, kind, inputs));
    }
    fn emit_diagnostic_impl(&mut self, diagnostic: Diagnostic) {
        if self.print_diagnostics {
            println!("{}", diagnostic.report()); // Allow println
        } else {
            self.diagnostics.insert(diagnostic);
        }
    }
    fn add_error(&mut self, span: impl Into<Span>, message: impl ToString) {
        let e = UiuaErrorKind::Run(
            span.into().sp(message.to_string()),
            self.asm.inputs.clone().into(),
        )
        .into();
        self.errors.push(e);
    }
    fn experimental_error<S>(&mut self, span: &CodeSpan, message: impl FnOnce() -> S)
    where
        S: ToString,
    {
        let take = self
            .scopes()
            .position(|sc| matches!(sc.kind, ScopeKind::File(_)))
            .map(|i| i + 1)
            .unwrap_or(usize::MAX);
        if !self
            .scopes()
            .take(take)
            .any(|sc| sc.experimental || sc.experimental_error)
        {
            self.scope.experimental_error = true;
            self.add_error(span.clone(), message().to_string());
        }
    }
    fn fatal_error(&self, span: impl Into<Span>, message: impl ToString) -> UiuaError {
        UiuaErrorKind::Run(
            span.into().sp(message.to_string()),
            self.asm.inputs.clone().into(),
        )
        .into()
    }
    fn validate_local(&mut self, name: &str, local: LocalName, span: &CodeSpan) {
        if local.public {
            return;
        }
        let get = |scope: &Scope| {
            (scope.names.get(name))
                .or_else(|| scope.names.get(name.strip_suffix('!')?))
                .copied()
        };
        if !local.public
            && get(&self.scope)
                .filter(|l| l.public || !matches!(self.scope.kind, ScopeKind::AllInModule))
                .or_else(|| {
                    self.higher_scopes
                        .last()
                        .filter(|_| !matches!(self.scope.kind, ScopeKind::File(_)))
                        .and_then(get)
                })
                .map_or(true, |l| l.index != local.index)
        {
            self.add_error(span.clone(), format!("`{}` is private", name));
        }
    }
    /// Get a span by its index
    pub fn get_span(&self, span: usize) -> Span {
        self.asm.spans[span].clone()
    }
    /// Register a span
    pub fn add_span(&mut self, span: impl Into<Span>) -> usize {
        let span = span.into();
        if let Some(i) = self.asm.spans.iter().position(|s| *s == span) {
            return i;
        }
        let idx = self.asm.spans.len();
        self.asm.spans.push(span);
        idx
    }
    /// Create a function
    pub fn create_function(
        &mut self,
        signature: impl Into<Signature>,
        f: impl Fn(&mut Uiua) -> UiuaResult + Send + Sync + 'static,
    ) -> Function {
        let signature = signature.into();
        let index = self.asm.dynamic_functions.len();
        self.asm.dynamic_functions.push(Arc::new(f));
        self.asm.add_function(
            FunctionId::Unnamed,
            signature,
            Node::Dynamic(DynamicFunction {
                index,
                sig: signature,
            }),
        )
    }
    /// Bind a function in the current scope
    ///
    /// # Errors
    /// Returns an error in the binding name is not valid
    pub fn bind_function(&mut self, name: impl Into<EcoString>, function: Function) -> UiuaResult {
        let index = self.next_global;
        let name = name.into();
        let local = LocalName {
            index,
            public: true,
        };
        self.next_global += 1;
        self.compile_bind_function(name.clone(), local, function, 0, None)?;
        Ok(())
    }
    /// Create and bind a function in the current scope
    ///
    /// # Errors
    /// Returns an error in the binding name is not valid
    pub fn create_bind_function(
        &mut self,
        name: impl Into<EcoString>,
        signature: impl Into<Signature>,
        f: impl Fn(&mut Uiua) -> UiuaResult + Send + Sync + 'static,
    ) -> UiuaResult {
        let function = self.create_function(signature, f);
        self.bind_function(name, function)
    }
    fn sig_of(&self, node: &Node, span: &CodeSpan) -> UiuaResult<Signature> {
        node.sig().map_err(|e| {
            self.fatal_error(
                span.clone(),
                format!("Cannot infer function signature: {e}"),
            )
        })
    }
}

fn words_look_pervasive(words: &[Sp<Word>]) -> bool {
    use Primitive::*;
    words.iter().all(|word| match &word.value {
        Word::Primitive(p) if p.class().is_pervasive() => true,
        Word::Primitive(Dup | Dip | Identity | Fork | Under | Each) => true,
        Word::Func(func) if func.lines.iter().all(|line| words_look_pervasive(line)) => true,
        Word::Number(..) | Word::Char(..) => true,
        Word::Modified(m) if m.modifier.value == Modifier::Primitive(Primitive::Each) => true,
        _ => false,
    })
}

fn collect_placeholder(words: &[Sp<Word>]) -> Vec<Sp<PlaceholderOp>> {
    let mut ops = Vec::new();
    for word in words {
        match &word.value {
            Word::Placeholder(op) => ops.push(word.span.clone().sp(*op)),
            Word::Strand(items) => ops.extend(collect_placeholder(items)),
            Word::Array(arr) => arr.lines.iter().for_each(|line| {
                ops.extend(collect_placeholder(line));
            }),
            Word::Func(func) => func.lines.iter().rev().for_each(|line| {
                ops.extend(collect_placeholder(line));
            }),
            Word::Modified(m) => ops.extend(collect_placeholder(&m.operands)),
            Word::Pack(pack) => pack.branches.iter().for_each(|branch| {
                (branch.value.lines.iter()).for_each(|line| ops.extend(collect_placeholder(line)))
            }),
            _ => {}
        }
    }
    ops
}

fn set_in_macro_arg(words: &mut Vec<Sp<Word>>) {
    recurse_words_mut(words, &mut |word| match &mut word.value {
        Word::Ref(r) => r.in_macro_arg = true,
        Word::IncompleteRef { in_macro_arg, .. } => *in_macro_arg = true,
        _ => {}
    });
}

fn recurse_words(words: &[Sp<Word>], f: &mut dyn FnMut(&Sp<Word>)) {
    for word in words {
        f(word);
        match &word.value {
            Word::Strand(items) => recurse_words(items, f),
            Word::Array(arr) => arr.lines.iter().for_each(|line| {
                recurse_words(line, f);
            }),
            Word::Func(func) => func.lines.iter().for_each(|line| {
                recurse_words(line, f);
            }),
            Word::Modified(m) => recurse_words(&m.operands, f),
            Word::Pack(pack) => pack.branches.iter().for_each(|branch| {
                (branch.value.lines.iter()).for_each(|line| recurse_words(line, f))
            }),
            _ => {}
        }
    }
}

fn recurse_words_mut(words: &mut Vec<Sp<Word>>, f: &mut dyn FnMut(&mut Sp<Word>)) {
    for word in words {
        match &mut word.value {
            Word::Strand(items) => recurse_words_mut(items, f),
            Word::Array(arr) => arr.lines.iter_mut().for_each(|line| {
                recurse_words_mut(line, f);
            }),
            Word::Func(func) => func.lines.iter_mut().for_each(|line| {
                recurse_words_mut(line, f);
            }),
            Word::Modified(m) => recurse_words_mut(&mut m.operands, f),
            Word::Pack(pack) => pack.branches.iter_mut().for_each(|branch| {
                (branch.value.lines.iter_mut()).for_each(|line| recurse_words_mut(line, f))
            }),
            _ => {}
        }
        f(word);
    }
}

fn line_sig(line: &[Sp<Word>]) -> Option<Sp<DocCommentSig>> {
    line.split_last()
        .and_then(|(last, mut rest)| match &last.value {
            Word::Comment(c) => c.parse::<DocCommentSig>().ok().map(|sig| {
                while rest.last().is_some_and(|w| matches!(w.value, Word::Spaces)) {
                    rest = &rest[..rest.len() - 1];
                }
                let span = (rest.first())
                    .zip(rest.last())
                    .map(|(f, l)| f.span.clone().merge(l.span.clone()))
                    .unwrap_or_else(|| last.span.clone());
                span.sp(sig)
            }),
            _ => None,
        })
}
