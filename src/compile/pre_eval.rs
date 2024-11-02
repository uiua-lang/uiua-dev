//! Pre-evaluate code at compile time

use std::time::Duration;

use indexmap::IndexSet;

use crate::check::nodes_clean_sig;

use super::*;

/// The mode that dictates how much code to pre-evaluate at compile time
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum PreEvalMode {
    /// Does not evalute pure constants and expressions at comptime, but still evaluates `comptime`
    Lazy,
    /// Evaluate as much as possible at compile time, even impure expressions
    ///
    /// Recursive functions and certain system functions are not evaluated
    Lsp,
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
        if let PreEvalMode::Lazy = self {
            return false;
        }
        if node.iter().any(|node| {
            matches!(
                node,
                Node::Push(val)
                    if val.element_count() > MAX_PRE_EVAL_ELEMS
                    || val.rank() > MAX_PRE_EVAL_RANK
            )
        }) || node.iter().all(|node| matches!(node, Node::Push(_)))
        {
            return false;
        }
        fn recurse<'a>(
            mode: PreEvalMode,
            node: &'a Node,
            asm: &'a Assembly,
            visited: &mut IndexSet<&'a Function>,
        ) -> bool {
            let len = visited.len();
            let matches = match node {
                Node::Run(nodes) => nodes.iter().all(|node| recurse(mode, node, asm, visited)),
                Node::Mod(_, args, _) | Node::ImplMod(_, args, _) => {
                    args.iter().all(|sn| recurse(mode, &sn.node, asm, visited))
                }
                Node::NoInline(_) => false,
                Node::Array { inner, .. } => recurse(mode, inner, asm, visited),
                Node::Call(func, _) => recurse(mode, &asm[func], asm, visited),
                node => {
                    node.is_limit_bounded(asm)
                        && match mode {
                            PreEvalMode::Lsp => node.is_pure(Purity::Impure, asm),
                            _ => node.is_pure(Purity::Pure, asm),
                        }
                }
            };
            visited.truncate(len);
            matches
        }
        recurse(*self, node, asm, &mut IndexSet::new())
    }
}

impl Compiler {
    pub(super) fn pre_eval(&self, node: &Node) -> Option<(Node, Vec<UiuaError>)> {
        let mut errors = Vec::new();
        if !self.pre_eval_mode.matches_node(node, &self.asm) {
            return if node
                .iter()
                .any(|node| self.pre_eval_mode.matches_node(node, &self.asm))
            {
                let node = node
                    .iter()
                    .map(|node| {
                        if let Some((new, errs)) = self.pre_eval(node) {
                            errors.extend(errs);
                            new
                        } else {
                            node.clone()
                        }
                    })
                    .collect();
                Some((node, errors))
            } else {
                None
            };
        }
        // println!("pre eval {:?}", node);
        let mut start = 0;
        let mut new: Option<Node> = None;
        let allow_error = node.is_pure(Purity::Pure, &self.asm);
        'start: while start < node.len() {
            for end in (start + 1..=node.len()).rev() {
                let section = &node[start..end];
                if section
                    .iter()
                    .all(|node| node.is_pure(Purity::Pure, &self.asm))
                    && nodes_clean_sig(section).is_some_and(|sig| sig.args == 0 && sig.outputs > 0)
                {
                    let mut success = false;
                    match self.comptime_node(section.into()) {
                        Ok(Some(values)) => {
                            // println!("section: {section:?}");
                            // println!("values: {values:?}");
                            for val in &values {
                                val.validate_shape();
                            }
                            let new = new.get_or_insert_with(|| node[..start].into());
                            new.extend(values.into_iter().map(Node::Push));
                            success = true;
                        }
                        Ok(None) => {}
                        Err(e) if !allow_error || e.is_fill || self.in_try => {}
                        Err(e) => {
                            // println!("error: {e:?}");
                            errors.push(e)
                        }
                    }
                    if !success {
                        if let Some(new) = &mut new {
                            new.extend(section.iter().cloned());
                        }
                    }
                    start = end;
                    continue 'start;
                }
            }
            if let Some(new) = &mut new {
                new.push(node[start].clone())
            }
            start += 1;
        }
        new.map(|new| (new, errors))
    }
    fn comptime_node(&self, node: Node) -> UiuaResult<Option<Vec<Value>>> {
        thread_local! {
            static CACHE: RefCell<HashMap<Node, Option<Vec<Value>>>> = RefCell::new(HashMap::new());
        }
        CACHE.with(|cache| {
            if let Some(stack) = cache.borrow_mut().get(&node) {
                return Ok(stack.clone());
            }
            let mut asm = self.asm.clone();
            asm.root = node;
            let mut env = if self.pre_eval_mode == PreEvalMode::Lsp {
                #[cfg(feature = "native_sys")]
                {
                    Uiua::with_native_sys()
                }
                #[cfg(not(feature = "native_sys"))]
                Uiua::with_safe_sys()
            } else {
                Uiua::with_safe_sys()
            }
            .with_execution_limit(Duration::from_millis(40));
            match env.run_asm(asm) {
                Ok(()) => {
                    let stack = env.take_stack();
                    let res = if stack.iter().any(|v| {
                        v.element_count() > MAX_PRE_EVAL_ELEMS || v.rank() > MAX_PRE_EVAL_RANK
                    }) {
                        None
                    } else {
                        Some(stack)
                    };
                    cache.borrow_mut().insert(env.asm.root, res.clone());
                    Ok(res)
                }
                Err(e) if matches!(e.kind, UiuaErrorKind::Timeout(..)) => {
                    cache.borrow_mut().insert(env.asm.root, None);
                    Ok(None)
                }
                Err(e) => Err(e),
            }
        })
    }
}
