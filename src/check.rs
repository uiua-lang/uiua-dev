//! Signature checker implementation

use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::HashMap,
    fmt,
    hash::{DefaultHasher, Hash, Hasher},
    iter::repeat,
    slice,
};

use enum_iterator::Sequence;

use crate::{Array, ImplPrimitive, Node, Primitive, SigNode, Signature, TempStack, Value};

impl Node {
    /// Get the signature of this node
    pub fn sig(&self) -> Result<Signature, SigCheckError> {
        VirtualEnv::from_node(self).map(|env| env.sig())
    }
    /// Get the signature of this node if there is no net temp stack change
    pub fn clean_sig(&self) -> Option<Signature> {
        nodes_clean_sig(slice::from_ref(self))
    }
    /// Get the signatures of this node on the temporary stacks
    pub fn temp_sigs(&self) -> Result<[Signature; TempStack::CARDINALITY], SigCheckError> {
        VirtualEnv::from_node(self).map(|env| env.temp_signatures())
    }
}

pub fn nodes_sig(nodes: &[Node]) -> Result<Signature, SigCheckError> {
    VirtualEnv::from_nodes(nodes).map(|env| env.sig())
}

pub fn nodes_all_sigs(nodes: &[Node]) -> Result<AllSignatures, SigCheckError> {
    type AllSigsCache = HashMap<u64, AllSignatures>;
    thread_local! {
        static CACHE: RefCell<AllSigsCache> = RefCell::new(AllSigsCache::new());
    }
    let mut hasher = DefaultHasher::new();
    nodes.hash(&mut hasher);
    let hash = hasher.finish();
    CACHE.with(|cache| {
        if let Some(sigs) = cache.borrow().get(&hash) {
            return Ok(*sigs);
        }
        let env = VirtualEnv::from_nodes(nodes)?;
        let sigs = AllSignatures {
            stack: env.sig(),
            temps: env.temp_signatures(),
        };
        cache.borrow_mut().insert(hash, sigs);
        Ok(sigs)
    })
}

pub fn nodes_clean_sig(nodes: &[Node]) -> Option<Signature> {
    let sigs = nodes_all_sigs(nodes).ok()?;
    if sigs.temps.iter().any(|&sig| sig != (0, 0)) {
        None
    } else {
        Some(sigs.stack)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AllSignatures {
    pub stack: Signature,
    pub temps: [Signature; TempStack::CARDINALITY],
}

pub(crate) fn naive_under_sig(f: Signature, g: Signature) -> Signature {
    let f_inv = if f.outputs > 1 {
        f.inverse()
    } else {
        Signature::new(f.args.min(1), f.outputs)
    };
    let mut curr = 0i32;
    let mut min = 0i32;
    curr -= f.args as i32;
    min = min.min(curr);
    curr += f.outputs as i32;
    curr -= g.args as i32;
    min = min.min(curr);
    curr += g.outputs as i32;
    curr -= f_inv.args as i32;
    min = min.min(curr);
    curr += f_inv.outputs as i32;
    Signature::new(min.unsigned_abs() as usize, (curr - min) as usize)
}

/// An environment that emulates the runtime but only keeps track of the stack.
struct VirtualEnv {
    stack: Vec<BasicValue>,
    height: i32,
    temp_stacks: [Vec<BasicValue>; TempStack::CARDINALITY],
    temp_heights: [i32; TempStack::CARDINALITY],
    min_height: usize,
    temp_min_heights: [usize; TempStack::CARDINALITY],
    array_depth: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SigCheckError {
    pub message: String,
    pub kind: SigCheckErrorKind,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum SigCheckErrorKind {
    Incorrect,
    Ambiguous,
    LoopOverreach,
    LoopVariable { sig: Signature, inf: bool },
}

impl SigCheckError {
    pub fn ambiguous(self) -> Self {
        Self {
            kind: SigCheckErrorKind::Ambiguous,
            ..self
        }
    }
    pub fn loop_overreach(self) -> Self {
        Self {
            kind: SigCheckErrorKind::LoopOverreach,
            ..self
        }
    }
    pub fn loop_variable(self, sig: Signature, inf: bool) -> Self {
        Self {
            kind: SigCheckErrorKind::LoopVariable { sig, inf },
            ..self
        }
    }
}

impl<'a> From<&'a str> for SigCheckError {
    fn from(s: &'a str) -> Self {
        Self {
            message: s.to_string(),
            kind: SigCheckErrorKind::Incorrect,
        }
    }
}

impl From<String> for SigCheckError {
    fn from(s: String) -> Self {
        Self {
            message: s,
            kind: SigCheckErrorKind::Incorrect,
        }
    }
}

impl fmt::Display for SigCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)
    }
}

#[derive(Debug, Clone)]
enum BasicValue {
    Num(f64),
    Arr(Vec<Self>),
    Other,
}

impl BasicValue {
    fn from_val(value: &Value) -> Self {
        if let Some(n) = value.as_num_array().and_then(Array::as_scalar) {
            BasicValue::Num(*n)
        } else if let Some(n) = value.as_byte_array().and_then(Array::as_scalar) {
            BasicValue::Num(*n as f64)
        } else if value.rank() == 1 {
            BasicValue::Arr(match value {
                Value::Num(n) => n.data.iter().map(|n| BasicValue::Num(*n)).collect(),
                Value::Byte(b) => b.data.iter().map(|b| BasicValue::Num(*b as f64)).collect(),
                Value::Complex(c) => c.data.iter().map(|_| BasicValue::Other).collect(),
                Value::Char(c) => c.data.iter().map(|_| BasicValue::Other).collect(),
                Value::Box(b) => b.data.iter().map(|_| BasicValue::Other).collect(),
            })
        } else {
            BasicValue::Other
        }
    }
}

impl FromIterator<f64> for BasicValue {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = f64>,
    {
        BasicValue::Arr(iter.into_iter().map(BasicValue::Num).collect())
    }
}

fn derive_sig(min_height: usize, final_height: i32) -> Signature {
    Signature {
        args: min_height,
        outputs: (final_height + min_height as i32).max(0) as usize,
    }
}

impl VirtualEnv {
    fn from_nodes(nodes: &[Node]) -> Result<Self, SigCheckError> {
        let mut env = VirtualEnv {
            stack: Vec::new(),
            height: 0,
            temp_stacks: Default::default(),
            temp_heights: Default::default(),
            min_height: 0,
            temp_min_heights: [0; TempStack::CARDINALITY],
            array_depth: 0,
        };
        env.nodes(nodes)?;
        Ok(env)
    }
    fn from_node(node: &Node) -> Result<Self, SigCheckError> {
        Self::from_nodes(slice::from_ref(node))
    }
    fn sig(&self) -> Signature {
        derive_sig(self.min_height, self.height)
    }
    fn temp_signatures(&self) -> [Signature; TempStack::CARDINALITY] {
        let mut sigs = [Signature::new(0, 0); TempStack::CARDINALITY];
        for ((sig, min_height), height) in sigs
            .iter_mut()
            .zip(&self.temp_min_heights)
            .zip(&self.temp_heights)
        {
            *sig = derive_sig(*min_height, *height);
        }
        sigs
    }
    fn nodes(&mut self, nodes: &[Node]) -> Result<(), SigCheckError> {
        nodes.iter().try_for_each(|node| self.node(node))
    }
    fn node(&mut self, node: &Node) -> Result<(), SigCheckError> {
        use Primitive::*;
        match node {
            Node::Run(nodes) => nodes.iter().try_for_each(|node| self.node(node))?,
            Node::Push(val) => self.push(BasicValue::from_val(val)),
            Node::Array { inner, .. } => {
                self.array_depth += 1;
                self.node(&inner.node)?;
                self.array_depth -= 1;
                let bottom = self.height - inner.sig.args as i32;
                let stack_bottom = (bottom.max(0) as usize).min(self.stack.len());
                let mut items: Vec<_> = (self.stack.drain(stack_bottom..))
                    .chain(repeat(BasicValue::Other).take((-bottom).max(0) as usize))
                    .collect();
                self.height = bottom;
                self.set_min_height();
                items.reverse();
                self.push(BasicValue::Arr(items));
            }
            Node::Label(..) | Node::RemoveLabel(_) => self.handle_args_outputs(1, 1),
            Node::Call(func, _) => self.handle_sig(func.sig()),
            Node::CallMacro(_, sig, _) | Node::CallGlobal(_, sig) => self.handle_sig(*sig),
            Node::BindGlobal { .. } => self.handle_args_outputs(1, 0),
            Node::CustomInverse(cust, _) => self.handle_sig(cust.normal.sig),
            Node::Dynamic(dy) => self.handle_sig(dy.sig),
            &Node::Switch {
                sig, under_cond, ..
            } => {
                let cond = self.pop();
                if under_cond {
                    self.push_temp(TempStack::Under, cond);
                }
                self.handle_sig(sig);
            }
            Node::Format(parts, ..) => self.handle_args_outputs(parts.len().saturating_sub(1), 1),
            Node::MatchFormatPattern(parts, ..) => {
                self.handle_args_outputs(1, parts.len().saturating_sub(1))
            }
            Node::Unpack { count, .. } => self.handle_args_outputs(1, *count),
            Node::Mod(Astar, args, _) | Node::ImplMod(ImplPrimitive::AstarFirst, args, _) => {
                let _start = self.pop();
                let [neighbors, heuristic, is_goal] = get_args(args)?;
                let args = neighbors
                    .args
                    .max(heuristic.args)
                    .max(is_goal.args)
                    .saturating_sub(1);
                self.handle_args_outputs(args, 2);
            }
            Node::Prim(prim, _) => match prim {
                Dup => {
                    let val = self.pop();
                    self.set_min_height();
                    self.push(val.clone());
                    self.push(val);
                }
                Flip => {
                    let a = self.pop();
                    let b = self.pop();
                    self.set_min_height();
                    self.push(a);
                    self.push(b);
                }
                Pop => {
                    self.pop();
                    self.set_min_height();
                }
                Over => {
                    let a = self.pop();
                    let b = self.pop();
                    self.set_min_height();
                    self.push(b.clone());
                    self.push(a);
                    self.push(b);
                }
                Around => {
                    let a = self.pop();
                    let b = self.pop();
                    self.set_min_height();
                    self.push(a.clone());
                    self.push(b);
                    self.push(a);
                }
                Join => {
                    let a = self.pop();
                    let b = self.pop();
                    self.set_min_height();
                    match (a, b) {
                        (BasicValue::Arr(mut a), BasicValue::Arr(b)) => {
                            a.extend(b);
                            self.push(BasicValue::Arr(a));
                        }
                        (BasicValue::Arr(mut a), b) => {
                            a.push(b);
                            self.push(BasicValue::Arr(a));
                        }
                        (a, BasicValue::Arr(mut b)) => {
                            b.insert(0, a);
                            self.push(BasicValue::Arr(b));
                        }
                        (a, b) => {
                            self.push(BasicValue::Arr(vec![a, b]));
                        }
                    }
                }
                prim => {
                    let args = prim
                        .args()
                        .ok_or_else(|| format!("{prim} has indeterminate args"))?;
                    let outputs = prim
                        .outputs()
                        .ok_or_else(|| format!("{prim} has indeterminate outputs"))?;
                    self.handle_args_outputs(args, outputs);
                }
            },
            Node::ImplPrim(prim, _) => {
                let args = prim.args();
                let outputs = prim.outputs();
                self.handle_args_outputs(args, outputs);
            }
            Node::Mod(prim, args, _) => match prim {
                Reduce | Scan => {
                    let [sig] = get_args(args)?;
                    let args = sig.args.saturating_sub(sig.outputs);
                    self.handle_args_outputs(args, sig.outputs);
                }
                Each | Rows | Inventory => {
                    let [sig] = get_args(args)?;
                    self.handle_sig(sig)
                }
                Table | Tuples | Triangle => {
                    let [sig] = get_args(args)?;
                    self.handle_sig(sig);
                }
                Group | Partition => {
                    let [sig] = get_args(args)?;
                    self.handle_args_outputs(2, sig.outputs);
                }
                Spawn | Pool => {
                    let [sig] = get_args(args)?;
                    self.handle_args_outputs(sig.args, 1);
                }
                Repeat => {
                    let [f] = get_args(args)?;
                    let n = self.pop();
                    self.repeat(f, n)?;
                }
                Do => {
                    let [body, cond] = get_args(args)?;
                    let copy_count = cond.args.saturating_sub(cond.outputs.saturating_sub(1));
                    let cond_sub_sig =
                        Signature::new(cond.args, (cond.outputs + copy_count).saturating_sub(1));
                    let comp_sig = body.compose(cond_sub_sig);
                    if comp_sig.args < comp_sig.outputs && self.array_depth == 0 {
                        return Err(SigCheckError::from(format!(
                            "do with a function with signature {comp_sig}"
                        ))
                        .loop_variable(comp_sig, false));
                    }
                    self.handle_args_outputs(
                        comp_sig.args,
                        comp_sig.outputs + cond_sub_sig.outputs.saturating_sub(cond.args),
                    );
                }
                Un => {
                    let [sig] = get_args(args)?;
                    self.handle_sig(sig.inverse());
                }
                Anti => {
                    let [sig] = get_args(args)?;
                    self.handle_sig(sig.anti().unwrap_or(sig));
                }
                Under => {
                    let [f, g] = get_args(args)?;
                    self.handle_sig(naive_under_sig(f, g));
                }
                Fold => {
                    let [f] = get_args(args)?;
                    self.handle_sig(f);
                }
                Try => {
                    let [f_sig, _handler_sig] = get_args(args)?;
                    self.handle_sig(f_sig);
                }
                Case => {
                    let [f] = get_args(args)?;
                    self.handle_sig(f);
                }
                Fill => {
                    let [fill, f] = get_args(args)?;
                    if fill.outputs > 0 {
                        self.handle_sig(fill);
                    }
                    self.handle_args_outputs(fill.outputs, 0);
                    self.handle_sig(f);
                }
                Content | Memo | Comptime => {
                    let [f] = get_args(args)?;
                    self.handle_sig(f);
                }
                Dump => {
                    let [_] = get_args(args)?;
                }
                _ => todo!(),
            },
            Node::ImplMod(prim, args, _) => match prim {
                ImplPrimitive::ReduceContent | ImplPrimitive::ReduceDepth(_) => {
                    let [sig] = get_args(args)?;
                    let args = sig.args.saturating_sub(sig.outputs);
                    self.handle_args_outputs(args, sig.outputs);
                }
                ImplPrimitive::RepeatWithInverse => {
                    let [f, inv] = get_args(args)?;
                    if f.inverse() != inv {
                        return Err(SigCheckError::from(
                            "repeat inverse does not have inverse signature",
                        )
                        .ambiguous());
                    }
                    let n = self.pop();
                    self.repeat(f, n)?;
                }
                ImplPrimitive::UnFill => {
                    let [fill, f] = get_args(args)?;
                    if fill.outputs > 0 {
                        self.handle_sig(fill);
                    }
                    self.handle_args_outputs(fill.outputs, 0);
                    self.handle_sig(f);
                }
                prim => {
                    let args = prim.args();
                    for _ in 0..args {
                        self.pop();
                    }
                    self.set_min_height();
                    let outputs = prim.outputs();
                    for _ in 0..outputs {
                        self.push(BasicValue::Other);
                    }
                }
            },
            Node::SetOutputComment { .. } => {}
            Node::ValidateType { .. } => self.handle_args_outputs(1, 1),
        }
        // println!("{instr:?} -> {}/{}", -(self.min_height as i32), self.height);
        Ok(())
    }
    // Simulate popping a value. Errors if the stack is empty, which means the function has too many args.
    fn pop(&mut self) -> BasicValue {
        self.height -= 1;
        self.set_min_height();
        self.stack.pop().unwrap_or(BasicValue::Other)
    }
    fn push(&mut self, val: BasicValue) {
        self.height += 1;
        self.stack.push(val);
    }
    fn _pop_temp(&mut self, stack: TempStack) -> BasicValue {
        self.temp_heights[stack as usize] -= 1;
        self.temp_min_heights[stack as usize] = self.temp_min_heights[stack as usize]
            .max((-self.temp_heights[stack as usize]).max(0) as usize);
        self.temp_stacks[stack as usize]
            .pop()
            .unwrap_or(BasicValue::Other)
    }
    fn push_temp(&mut self, stack: TempStack, val: BasicValue) {
        self.temp_heights[stack as usize] += 1;
        self.temp_stacks[stack as usize].push(val);
    }
    /// Set the current stack height as a potential minimum.
    /// At the end of checking, the minimum stack height is a component in calculating the signature.
    fn set_min_height(&mut self) {
        self.min_height = self.min_height.max((-self.height).max(0) as usize);
        for (min_height, height) in self.temp_min_heights.iter_mut().zip(&self.temp_heights) {
            *min_height = (*min_height).max((-*height).max(0) as usize);
        }
    }
    fn handle_args_outputs(&mut self, args: usize, outputs: usize) {
        for _ in 0..args {
            self.pop();
        }
        for _ in 0..outputs {
            self.push(BasicValue::Other);
        }
    }
    fn handle_sig(&mut self, sig: Signature) {
        self.handle_args_outputs(sig.args, sig.outputs)
    }
    fn repeat(&mut self, sig: Signature, n: BasicValue) -> Result<(), SigCheckError> {
        if let BasicValue::Num(n) = n {
            // If n is a known natural number, then the function can have any signature.
            let sig = if n >= 0.0 { sig } else { sig.inverse() };
            if n.fract() == 0.0 {
                let n = n.abs() as usize;
                if n > 0 {
                    let (args, outputs) = match sig.args.cmp(&sig.outputs) {
                        Ordering::Equal => (sig.args, sig.outputs),
                        Ordering::Less => (sig.args, n * (sig.outputs - sig.args) + sig.args),
                        Ordering::Greater => {
                            ((n - 1) * (sig.args - sig.outputs) + sig.args, sig.outputs)
                        }
                    };
                    self.handle_args_outputs(args, outputs);
                }
            } else if n.is_infinite() {
                match sig.args.cmp(&sig.outputs) {
                    Ordering::Greater => {
                        return Err(SigCheckError::from(format!(
                            "repeat with infinity and a function with signature {sig}"
                        ))
                        .loop_overreach());
                    }
                    Ordering::Less if self.array_depth == 0 => {
                        return Err(SigCheckError::from(format!(
                            "repeat with infinity and a function with signature {sig}"
                        ))
                        .loop_variable(sig, true));
                    }
                    _ => self.handle_sig(sig),
                }
            } else {
                return Err("repeat without an integer or infinity".into());
            }
        } else {
            // If n is unknown, then what we do depends on the signature
            match sig.args.cmp(&sig.outputs) {
                Ordering::Equal => self.handle_sig(sig),
                Ordering::Greater => {
                    return Err(SigCheckError::from(format!(
                        "repeat with no number and a function with signature {sig}"
                    ))
                    .loop_overreach());
                }
                Ordering::Less if self.array_depth == 0 => {
                    return Err(SigCheckError::from(format!(
                        "repeat with no number and a function with signature {sig}"
                    ))
                    .loop_variable(sig, false));
                }
                Ordering::Less => self.handle_sig(sig),
            }
        }
        Ok(())
    }
}

fn get_args<const N: usize>(args: &[SigNode]) -> Result<[Signature; N], SigCheckError> {
    let mut res = [Signature::default(); N];
    if args.len() != N {
        return Err(format!("Expected {} arguments, but got {}", N, args.len()).into());
    }
    for (i, arg) in args.iter().enumerate() {
        res[i] = arg.sig;
    }
    Ok(res)
}
