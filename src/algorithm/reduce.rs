//! Algorithms for reducing modifiers

use std::{collections::VecDeque, convert::identity, iter::repeat};

use ecow::{eco_vec, EcoVec};

use crate::{
    algorithm::{get_ops, loops::flip, multi_output, pervade::*},
    check::nodes_sig,
    cowslice::cowslice,
    Array, ArrayValue, Complex, ImplPrimitive, Node, Ops, Primitive, Shape, SigNode, Signature,
    Uiua, UiuaResult, Value,
};

use super::{fixed_rows, FillContext, FixedRowsData};

pub fn reduce(ops: Ops, depth: usize, env: &mut Uiua) -> UiuaResult {
    crate::profile_function!();
    let [f] = get_ops(ops, env)?;
    reduce_impl(f, depth, env)
}

pub(crate) fn reduce_impl(f: SigNode, depth: usize, env: &mut Uiua) -> UiuaResult {
    if f.sig.args < 2 {
        return Err(env.error(format!(
            "{}'s function must have at least 2 arguments, \
            but its signature is {}",
            Primitive::Reduce.format(),
            f.sig
        )));
    }
    let xs = env.pop(1)?;
    match (f.node.as_flipped_primitive(), xs) {
        (Some((Primitive::Join, false)), mut xs)
            if env.value_fill().is_none() && env.value_fill().is_none() =>
        {
            let depth = depth.min(xs.rank());
            if xs.rank() - depth < 2 {
                env.push(xs);
                return Ok(());
            }
            let shape = xs.shape();
            let mut new_shape = Shape::with_capacity(xs.rank() - 1);
            new_shape.extend_from_slice(&shape[..depth]);
            new_shape.push(shape[depth] * shape[depth + 1]);
            new_shape.extend_from_slice(&shape[depth + 2..]);
            *xs.shape_mut() = new_shape;
            env.push(xs);
        }
        (Some((prim, flipped)), Value::Num(nums)) => {
            if let Err(nums) = reduce_nums(prim, flipped, nums, depth, env) {
                return generic_reduce(f, Value::Num(nums), depth, env);
            }
        }

        (Some((prim, flipped)), Value::Complex(nums)) => {
            if let Err(nums) = reduce_coms(prim, flipped, nums, depth, env) {
                return generic_reduce(f, Value::Complex(nums), depth, env);
            }
        }
        (Some((prim, flipped)), Value::Byte(bytes)) => {
            let fill = env.scalar_fill::<f64>().ok();
            env.push::<Value>(match prim {
                Primitive::Add => {
                    fast_reduce_different(bytes, 0.0, fill, depth, add::num_num, add::num_byte)
                        .into()
                }
                Primitive::Sub if flipped => fast_reduce_different(
                    bytes,
                    0.0,
                    fill,
                    depth,
                    flip(sub::num_num),
                    flip(sub::byte_num),
                )
                .into(),
                Primitive::Sub => {
                    fast_reduce_different(bytes, 0.0, fill, depth, sub::num_num, sub::num_byte)
                        .into()
                }
                Primitive::Mul if bytes.meta().flags.is_boolean() => {
                    let byte_fill = env.scalar_fill::<u8>().ok();
                    if bytes.row_count() == 0 || fill.is_some() && byte_fill.is_none() {
                        fast_reduce_different(bytes, 1.0, fill, depth, mul::num_num, mul::num_byte)
                            .into()
                    } else {
                        fast_reduce(bytes, 1, byte_fill, depth, mul::bool_bool).into()
                    }
                }
                Primitive::Mul => {
                    fast_reduce_different(bytes, 1.0, fill, depth, mul::num_num, mul::num_byte)
                        .into()
                }
                Primitive::Div if flipped => fast_reduce_different(
                    bytes,
                    1.0,
                    fill,
                    depth,
                    flip(div::num_num),
                    flip(div::byte_num),
                )
                .into(),
                Primitive::Div => {
                    fast_reduce_different(bytes, 1.0, fill, depth, div::num_num, div::num_byte)
                        .into()
                }
                Primitive::Modulus if flipped => fast_reduce_different(
                    bytes,
                    1.0,
                    fill,
                    depth,
                    flip(modulus::num_num),
                    flip(modulus::byte_num),
                )
                .into(),
                Primitive::Modulus => fast_reduce_different(
                    bytes,
                    1.0,
                    fill,
                    depth,
                    modulus::num_num,
                    modulus::num_byte,
                )
                .into(),
                Primitive::Atan if flipped => fast_reduce_different(
                    bytes,
                    0.0,
                    fill,
                    depth,
                    flip(atan2::num_num),
                    flip(atan2::byte_num),
                )
                .into(),
                Primitive::Atan => {
                    fast_reduce_different(bytes, 0.0, fill, depth, atan2::num_num, atan2::num_byte)
                        .into()
                }
                Primitive::Max => {
                    let byte_fill = env.scalar_fill::<u8>().ok();
                    if bytes.row_count() == 0 || fill.is_some() && byte_fill.is_none() {
                        fast_reduce_different(
                            bytes,
                            f64::NEG_INFINITY,
                            fill,
                            depth,
                            max::num_num,
                            max::num_byte,
                        )
                        .into()
                    } else {
                        fast_reduce(bytes, 0, byte_fill, depth, max::byte_byte).into()
                    }
                }
                Primitive::Min => {
                    let byte_fill = env.scalar_fill::<u8>().ok();
                    if bytes.row_count() == 0 || fill.is_some() && byte_fill.is_none() {
                        fast_reduce_different(
                            bytes,
                            f64::INFINITY,
                            fill,
                            depth,
                            min::num_num,
                            min::num_byte,
                        )
                        .into()
                    } else {
                        fast_reduce(bytes, 0, byte_fill, depth, min::byte_byte).into()
                    }
                }
                _ => return generic_reduce(f, Value::Byte(bytes), depth, env),
            })
        }
        (_, xs) if f.sig == (2, 1) => {
            if depth == 0 && env.value_fill().is_none() {
                if xs.row_count() == 0 {
                    let val = reduce_identity(&f.node, xs).ok_or_else(|| {
                        env.error(format!(
                            "Cannot {} empty array. Function has no identity value.",
                            Primitive::Reduce.format()
                        ))
                    })?;
                    env.push(val);
                    return Ok(());
                }
                if xs.row_count() == 1 {
                    let val = reduce_singleton(&f.node, xs, identity);
                    env.push(val);
                    return Ok(());
                }
            }
            generic_reduce(f, xs, depth, env)?
        }
        (_, xs) => generic_reduce(f, xs, depth, env)?,
    }
    Ok(())
}

fn trim_node(node: &Node) -> &[Node] {
    let mut nodes = node.as_slice();
    use ImplPrimitive::*;
    use Primitive::*;
    let trim = |node: &Node| {
        matches!(
            node,
            Node::Prim(Stack | Trace, _) | Node::ImplPrim(UnStack | TraceN { .. }, _)
        )
    };
    while nodes.first().is_some_and(trim) {
        nodes = &nodes[1..];
    }
    while nodes.last().is_some_and(trim) {
        nodes = &nodes[..nodes.len() - 1];
    }
    nodes
}

fn reduce_identity(node: &Node, mut val: Value) -> Option<Value> {
    use Primitive::*;
    let nodes = trim_node(node);
    let mut shape = val.shape().clone();
    shape.make_row();
    let len: usize = shape.iter().product();
    let (first, tail) = nodes.split_first()?;
    let (last, init) = nodes.split_last()?;
    let init_sig = || nodes_sig(init).is_ok_and(|sig| sig.args == sig.outputs);
    let tail_sig = || nodes_sig(tail).is_ok_and(|sig| sig.args == 1 && sig.outputs == 1);
    Some(match first {
        Node::Prim(Join, _) if tail_sig() => {
            if val.rank() < 2 {
                val.shape_mut()[0] = 0;
            } else {
                let first = val.shape_mut().remove(0);
                val.shape_mut()[0] *= first;
            }
            val
        }
        _ => match last {
            Node::Prim(Add | Sub, _) if init_sig() => Array::new(shape, eco_vec![0u8; len]).into(),
            Node::Prim(Mul | Div | Modulus, _) if init_sig() => {
                Array::new(shape, eco_vec![1u8; len]).into()
            }
            Node::Prim(Max, _) if init_sig() => {
                Array::new(shape, eco_vec![f64::NEG_INFINITY; len]).into()
            }
            Node::Prim(Min, _) if init_sig() => {
                Array::new(shape, eco_vec![f64::INFINITY; len]).into()
            }
            Node::Prim(Atan, _) if init_sig() => Array::new(shape, eco_vec![0.0; len]).into(),
            Node::Prim(Join, _) if init_sig() => {
                if val.rank() < 2 {
                    val.shape_mut()[0] = 0;
                } else {
                    let first = val.shape_mut().remove(0);
                    val.shape_mut()[0] *= first;
                }
                val
            }
            Node::Format(parts, _) if parts.len() == 3 && init_sig() => {
                EcoVec::<char>::new().into()
            }
            _ => return None,
        },
    })
}

fn reduce_singleton(node: &Node, val: Value, process: impl Fn(Value) -> Value) -> Value {
    use Primitive::*;
    let nodes = trim_node(node);
    let row = process(val.row(0));
    let Some((first, tail)) = nodes.split_first() else {
        return val;
    };
    let (last, init) = nodes.split_last().unwrap();
    let init_sig = || nodes_sig(init).is_ok_and(|sig| sig.args == sig.outputs);
    let tail_sig = || nodes_sig(tail).is_ok_and(|sig| sig.args == 1 && sig.outputs == 1);
    match first {
        Node::Prim(Join, _) if tail_sig() => val,
        _ => match last {
            Node::Prim(Join, _) if init_sig() => {
                if val.rank() < 2 {
                    val
                } else {
                    row
                }
            }
            Node::Format(parts, _) if init_sig() && parts.len() == 3 => row.format().into(),
            _ => row,
        },
    }
}

macro_rules! reduce_math {
    ($fname:ident, $ty:ident, $f:ident) => {
        #[allow(clippy::result_large_err)]
        fn $fname(
            prim: Primitive,
            _flipped: bool,
            xs: Array<$ty>,
            depth: usize,
            env: &mut Uiua,
        ) -> Result<(), Array<$ty>>
        where
            $ty: From<f64>,
        {
            let fill = env.scalar_fill::<$ty>().ok();
            env.push(match prim {
                Primitive::Add => fast_reduce(xs, 0.0.into(), fill, depth, add::$f),
                #[cfg(feature = "opt")]
                Primitive::Sub if _flipped => {
                    fast_reduce(xs, 0.0.into(), fill, depth, flip(sub::$f))
                }
                #[cfg(feature = "opt")]
                Primitive::Sub => fast_reduce(xs, 0.0.into(), fill, depth, sub::$f),
                Primitive::Mul => fast_reduce(xs, 1.0.into(), fill, depth, mul::$f),
                #[cfg(feature = "opt")]
                Primitive::Div if _flipped => {
                    fast_reduce(xs, 1.0.into(), fill, depth, flip(div::$f))
                }
                #[cfg(feature = "opt")]
                Primitive::Div => fast_reduce(xs, 1.0.into(), fill, depth, div::$f),
                #[cfg(feature = "opt")]
                Primitive::Modulus if _flipped => {
                    fast_reduce(xs, 1.0.into(), fill, depth, flip(modulus::$f))
                }
                #[cfg(feature = "opt")]
                Primitive::Modulus => fast_reduce(xs, 1.0.into(), fill, depth, modulus::$f),
                #[cfg(feature = "opt")]
                Primitive::Atan if _flipped => {
                    fast_reduce(xs, 0.0.into(), fill, depth, flip(atan2::$f))
                }
                #[cfg(feature = "opt")]
                Primitive::Atan => fast_reduce(xs, 0.0.into(), fill, depth, atan2::$f),
                Primitive::Max => fast_reduce(xs, f64::NEG_INFINITY.into(), fill, depth, max::$f),
                Primitive::Min => fast_reduce(xs, f64::INFINITY.into(), fill, depth, min::$f),
                _ => return Err(xs),
            });
            Ok(())
        }
    };
}

reduce_math!(reduce_nums, f64, num_num);
reduce_math!(reduce_coms, Complex, com_x);

fn fast_reduce_different<T, U>(
    arr: Array<T>,
    identity: U,
    default: Option<U>,
    mut depth: usize,
    fuu: impl Fn(U, U) -> U,
    fut: impl Fn(U, T) -> U,
) -> Array<U>
where
    T: ArrayValue + Copy + Into<U>,
    U: ArrayValue + Copy,
{
    depth = depth.min(arr.rank());
    if depth == 0 && arr.rank() == 1 {
        return if let Some(default) = default {
            arr.data.into_iter().fold(default, fut).into()
        } else if arr.row_count() == 0 {
            identity.into()
        } else {
            let first = arr.data[0].into();
            arr.data[1..].iter().copied().fold(first, fut).into()
        };
    }
    fast_reduce(arr.convert(), identity, default, depth, fuu)
}

fn fast_reduce<T>(
    mut arr: Array<T>,
    identity: T,
    default: Option<T>,
    mut depth: usize,
    f: impl Fn(T, T) -> T,
) -> Array<T>
where
    T: ArrayValue + Copy,
{
    depth = depth.min(arr.rank());
    if depth == 0 && arr.rank() == 1 {
        return if let Some(default) = default {
            arr.data.iter().copied().fold(default, f).into()
        } else if arr.row_count() == 0 {
            identity.into()
        } else {
            let first = arr.data[0];
            arr.data[1..].iter().copied().fold(first, f).into()
        };
    }
    match (arr.rank(), depth) {
        (r, d) if r == d => arr,
        (1, 0) => {
            let data = arr.data.as_mut_slice();
            let reduced = default.into_iter().chain(data.iter().copied()).reduce(f);
            if let Some(reduced) = reduced {
                if data.is_empty() {
                    arr.data.extend(Some(reduced));
                } else {
                    data[0] = reduced;
                    arr.data.truncate(1);
                }
            } else {
                arr.data.extend(Some(identity));
            }
            arr.shape = Shape::default();
            arr.validate_shape();
            arr
        }
        (_, 0) => {
            let row_len = arr.row_len();
            if row_len == 0 {
                arr.shape.remove(0);
                return Array::new(arr.shape, EcoVec::new());
            }
            let row_count = arr.row_count();
            if row_count == 0 {
                arr.shape.remove(0);
                let data = cowslice![default.unwrap_or(identity); row_len];
                return Array::new(arr.shape, data);
            }
            let sliced = arr.data.as_mut_slice();
            let (acc, rest) = sliced.split_at_mut(row_len);
            if let Some(default) = default {
                for acc in &mut *acc {
                    *acc = f(default, *acc);
                }
            }
            rest.chunks_exact(row_len).fold(acc, |acc, row| {
                for (a, b) in acc.iter_mut().zip(row) {
                    *a = f(*a, *b);
                }
                acc
            });
            arr.data.truncate(row_len);
            arr.shape.remove(0);
            arr.validate_shape();
            arr
        }
        (_, depth) => {
            let chunk_count: usize = arr.shape[..depth].iter().product();
            let chunk_len: usize = arr.shape[depth..].iter().product();
            let chunk_row_len: usize = arr.shape[depth + 1..].iter().product();
            let data_slice = arr.data.as_mut_slice();
            if chunk_len == 0 {
                let val = default.unwrap_or(identity);
                arr.data = repeat(val).take(chunk_count * chunk_row_len).collect();
            } else {
                for c in 0..chunk_count {
                    let chunk_start = c * chunk_len;
                    let chunk = &mut data_slice[chunk_start..][..chunk_len];
                    let (acc, rest) = chunk.split_at_mut(chunk_row_len);
                    if let Some(default) = default {
                        for acc in &mut *acc {
                            *acc = f(default, *acc);
                        }
                    }
                    rest.chunks_exact_mut(chunk_row_len).fold(acc, |acc, row| {
                        for (a, b) in acc.iter_mut().zip(row) {
                            *a = f(*a, *b);
                        }
                        acc
                    });
                    data_slice
                        .copy_within(chunk_start..chunk_start + chunk_row_len, c * chunk_row_len);
                }
                arr.data.truncate(chunk_count * chunk_row_len);
            }
            arr.shape.remove(depth);
            arr.validate_shape();
            arr
        }
    }
}

fn generic_reduce(f: SigNode, xs: Value, depth: usize, env: &mut Uiua) -> UiuaResult {
    env.push(xs);
    let val = generic_reduce_inner(f, depth, identity, env)?;
    env.push(val);
    Ok(())
}

pub fn reduce_content(ops: Ops, env: &mut Uiua) -> UiuaResult {
    let [f] = get_ops(ops, env)?;
    let xs = env.pop(1)?;
    if let (1, Some((Primitive::Join, false))) = (xs.rank(), f.node.as_flipped_primitive()) {
        if xs.row_count() == 0 {
            env.push(match xs {
                Value::Box(_) => Value::default(),
                value => value,
            });
            return Ok(());
        }
        let mut rows = xs.into_rows().map(Value::unboxed);
        let mut acc = rows.next().unwrap();
        if acc.rank() == 0 {
            acc.shape_mut().insert(0, 1);
        }
        for row in rows {
            acc = acc.join(row, true, env)?;
        }
        env.push(acc);
        return Ok(());
    }
    env.push(xs);
    let val = generic_reduce_inner(f, 0, Value::unboxed, env)?;
    env.push(val);
    Ok(())
}

fn generic_reduce_inner(
    f: SigNode,
    depth: usize,
    process: impl Fn(Value) -> Value + Copy,
    env: &mut Uiua,
) -> UiuaResult<Value> {
    let sig = f.sig;
    if sig.outputs != 1 {
        return Err(env.error(format!(
            "{}'s function must have exactly 1 output, \
            but its signature is {sig}",
            Primitive::Reduce.format(),
        )));
    }
    match sig.args {
        0 | 1 => Err(env.error(format!(
            "{}'s function must have at least 2 arguments, \
            but its signature is {sig}",
            Primitive::Reduce.format(),
        ))),
        2 => {
            let mut xs = env.pop(1)?;
            if depth == 0 {
                let value_fill = env.value_fill();
                if value_fill.is_none() {
                    if xs.row_count() == 0 {
                        return reduce_identity(&f.node, xs).ok_or_else(|| {
                            env.error(format!(
                                "Cannot {} empty array. Function has no identity value.",
                                Primitive::Reduce.format()
                            ))
                        });
                    }
                    if xs.row_count() == 1 {
                        let row_count = if xs.rank() == 0 {
                            None
                        } else {
                            Some(xs.shape_mut().remove(0))
                        };
                        if let Some(row_count) = row_count {
                            xs.shape_mut().insert(0, row_count);
                        }
                        return Ok(reduce_singleton(&f.node, xs, process));
                    }
                }
                let mut rows = xs.into_rows();
                let mut acc = (value_fill.cloned())
                    .or_else(|| rows.next())
                    .ok_or_else(|| {
                        env.error(format!("Cannot {} empty array", Primitive::Reduce.format()))
                    })?;
                acc = process(acc);
                env.without_fill(|env| -> UiuaResult<Value> {
                    for row in rows {
                        env.push(process(row));
                        env.push(acc);
                        env.exec(f.clone())?;
                        acc = env.pop("reduced function result")?;
                    }
                    Ok(acc)
                })
            } else {
                let mut new_rows = Vec::with_capacity(xs.row_count());
                env.without_fill(|env| -> UiuaResult {
                    for row in xs.into_rows() {
                        env.push(row);
                        let val = generic_reduce_inner(f.clone(), depth - 1, process, env)?;
                        new_rows.push(val);
                    }
                    Ok(())
                })?;
                Value::from_row_values(new_rows, env)
            }
        }
        n => {
            let mut repeated = Vec::with_capacity(n - 2);
            for i in 0..n - 2 {
                repeated.push(process(env.pop(i + 1)?));
            }
            let xs = env.pop(n - 1)?;
            if depth == 0 {
                let mut rows = xs.into_rows();
                let mut acc = (env.value_fill().cloned())
                    .or_else(|| rows.next())
                    .ok_or_else(|| {
                        env.error(format!("Cannot {} empty array", Primitive::Reduce.format()))
                    })?;
                acc = process(acc);
                env.without_fill(|env| {
                    for row in rows {
                        env.push(process(row));
                        for val in repeated.iter().rev() {
                            env.push(val.clone());
                        }
                        env.push(acc);
                        env.exec(f.clone())?;
                        acc = env.pop("reduced function result")?;
                    }
                    Ok(acc)
                })
            } else {
                let mut new_values = Vec::with_capacity(xs.row_count());
                let mut args = repeated;
                args.push(xs);
                let FixedRowsData {
                    row_count,
                    mut rows,
                    all_scalar,
                    is_empty,
                    per_meta,
                    ..
                } = fixed_rows(Primitive::Rows.format(), 1, args, env)?;
                env.without_fill(|env| -> UiuaResult {
                    for _ in 0..row_count {
                        for arg in rows.iter_mut().rev() {
                            match arg {
                                Ok(rows) => env.push(rows.next().unwrap()),
                                Err(row) => env.push(row.clone()),
                            }
                        }
                        let val = generic_reduce_inner(f.clone(), depth - 1, process, env)?;
                        new_values.push(val);
                    }
                    Ok(())
                })?;
                let mut rowsed = Value::from_row_values(new_values, env)?;
                if all_scalar {
                    rowsed.undo_fix();
                } else if is_empty {
                    rowsed.pop_row();
                }
                rowsed.validate_shape();
                rowsed.set_per_meta(per_meta.clone());
                Ok(rowsed)
            }
        }
    }
}

pub fn scan(ops: Ops, env: &mut Uiua) -> UiuaResult {
    crate::profile_function!();
    let [f] = get_ops(ops, env)?;
    let xs = env.pop(1)?;
    if xs.rank() == 0 && f.sig.args <= 2 {
        return Err(env.error(format!("Cannot {} rank 0 array", Primitive::Scan.format())));
    }
    match (f.node.as_flipped_primitive(), xs) {
        (Some((prim, flipped)), Value::Num(nums)) => {
            let arr = match prim {
                Primitive::Eq => fast_scan(nums, |a, b| is_eq::num_num(a, b) as f64),
                Primitive::Ne => fast_scan(nums, |a, b| is_ne::num_num(a, b) as f64),
                Primitive::Add => fast_scan(nums, add::num_num),
                Primitive::Sub if flipped => fast_scan(nums, flip(sub::num_num)),
                Primitive::Sub => fast_scan(nums, sub::num_num),
                Primitive::Mul => fast_scan(nums, mul::num_num),
                Primitive::Div if flipped => fast_scan(nums, flip(div::num_num)),
                Primitive::Div => fast_scan(nums, div::num_num),
                Primitive::Modulus if flipped => fast_scan(nums, flip(modulus::num_num)),
                Primitive::Modulus => fast_scan(nums, modulus::num_num),
                Primitive::Atan if flipped => fast_scan(nums, flip(atan2::num_num)),
                Primitive::Atan => fast_scan(nums, atan2::num_num),
                Primitive::Max => fast_scan(nums, max::num_num),
                Primitive::Min => fast_scan(nums, min::num_num),
                _ => return generic_scan(f, Value::Num(nums), env),
            };
            env.push(arr);
            Ok(())
        }
        (Some((prim, flipped)), Value::Byte(bytes)) => {
            match prim {
                Primitive::Eq => env.push(fast_scan(bytes, is_eq::generic)),
                Primitive::Ne => env.push(fast_scan(bytes, is_ne::generic)),
                Primitive::Add => env.push(fast_scan::<f64>(bytes.convert(), add::num_num)),
                Primitive::Sub if flipped => {
                    env.push(fast_scan::<f64>(bytes.convert(), flip(sub::num_num)))
                }
                Primitive::Sub => env.push(fast_scan::<f64>(bytes.convert(), sub::num_num)),
                Primitive::Mul => env.push(fast_scan::<f64>(bytes.convert(), mul::num_num)),
                Primitive::Div if flipped => {
                    env.push(fast_scan::<f64>(bytes.convert(), flip(div::num_num)))
                }
                Primitive::Div => env.push(fast_scan::<f64>(bytes.convert(), div::num_num)),
                Primitive::Modulus if flipped => {
                    env.push(fast_scan::<f64>(bytes.convert(), flip(modulus::num_num)))
                }
                Primitive::Modulus => env.push(fast_scan::<f64>(bytes.convert(), modulus::num_num)),
                Primitive::Atan if flipped => {
                    env.push(fast_scan::<f64>(bytes.convert(), flip(atan2::num_num)))
                }
                Primitive::Atan => env.push(fast_scan::<f64>(bytes.convert(), atan2::num_num)),
                Primitive::Max => env.push(fast_scan(bytes, u8::max)),
                Primitive::Min => env.push(fast_scan(bytes, u8::min)),
                _ => return generic_scan(f, Value::Byte(bytes), env),
            }
            Ok(())
        }
        (_, xs) => generic_scan(f, xs, env),
    }
}

fn fast_scan<T>(mut arr: Array<T>, f: impl Fn(T, T) -> T) -> Array<T>
where
    T: ArrayValue + Copy,
{
    match arr.shape.len() {
        0 => unreachable!("fast_scan called on unit array, should have been guarded against"),
        1 => {
            if arr.row_count() == 0 {
                return arr;
            }
            let mut acc = arr.data[0];
            for val in arr.data.as_mut_slice()[1..].iter_mut() {
                acc = f(acc, *val);
                *val = acc;
            }
            arr
        }
        _ => {
            let row_len: usize = arr.row_len();
            if arr.row_count() == 0 {
                return arr;
            }
            let shape = arr.shape.clone();
            let mut new_data = EcoVec::with_capacity(arr.data.len());
            let mut rows = arr.into_rows();
            new_data.extend(rows.next().unwrap().data);
            for row in rows {
                let start = new_data.len() - row_len;
                for (i, r) in row.data.into_iter().enumerate() {
                    new_data.push(f(new_data[start + i], r));
                }
            }
            Array::new(shape, new_data)
        }
    }
}

fn generic_scan(f: SigNode, xs: Value, env: &mut Uiua) -> UiuaResult {
    let sig = f.sig;
    if sig.outputs != 1 {
        return Err(env.error(format!(
            "{}'s function must have 1 output, \
            but its signature is {sig}",
            Primitive::Scan.format(),
        )));
    }
    match sig.args {
        0 | 1 => Err(env.error(format!(
            "{}'s function must have at least 2 arguments, \
            but its signature is {sig}",
            Primitive::Scan.format(),
        ))),
        2 => {
            if xs.row_count() == 0 {
                env.push(xs.first_dim_zero());
                return Ok(());
            }
            let row_count = xs.row_count();
            let mut rows = xs.into_rows();
            let mut acc = rows.next().unwrap();
            let mut scanned = Vec::with_capacity(row_count);
            scanned.push(acc.clone());
            env.without_fill(|env| -> UiuaResult {
                for row in rows.by_ref() {
                    env.push(row);
                    env.push(acc.clone());
                    env.exec(f.clone())?;
                    acc = env.pop("scanned function result")?;
                    scanned.push(acc.clone());
                }
                Ok(())
            })?;
            let val = Value::from_row_values(scanned, env)?;
            env.push(val);
            Ok(())
        }
        n => {
            let mut repeated = Vec::with_capacity(n - 1);
            repeated.push(xs);
            for i in 0..n - 2 {
                repeated.push(env.pop(i + 2)?);
            }
            let xs = repeated.pop().unwrap();
            if xs.row_count() == 0 {
                let val = reduce_identity(&f.node, xs.clone())
                    .map(|v| v.first_dim_zero())
                    .unwrap_or(xs);
                env.push(val);
                return Ok(());
            }
            let mut scanned = Vec::with_capacity(xs.row_count());
            let mut rows = xs.into_rows();
            let mut acc = rows.next().unwrap();
            scanned.push(acc.clone());
            env.without_fill(|env| -> UiuaResult {
                for row in rows {
                    env.push(row);
                    for val in repeated.iter().rev() {
                        env.push(val.clone());
                    }
                    env.push(acc);
                    env.exec(f.clone())?;
                    acc = env.pop("reduced function result")?;
                    scanned.push(acc.clone());
                }
                Ok(())
            })?;
            let val = Value::from_row_values(scanned, env)?;
            env.push(val);
            Ok(())
        }
    }
}

pub fn unscan(ops: Ops, env: &mut Uiua) -> UiuaResult {
    let [f] = get_ops(ops, env)?;
    let mut xs = env.pop(1)?;
    if xs.rank() == 0 {
        return Err(env.error(format!("Cannot {} rank 0 array", ImplPrimitive::UnScan,)));
    }
    let sig = f.sig;
    if sig != (2, 1) {
        return Err(env.error(format!(
            "{} unscan's function's signature must be |2.1, but it is {sig}",
            ImplPrimitive::UnScan,
        )));
    }
    if xs.row_count() == 0 {
        env.push(xs.first_dim_zero());
        return Ok(());
    }

    match xs {
        Value::Num(nums) => match f.node.as_flipped_primitive() {
            Some((Primitive::Sub, false)) => {
                env.push(fast_invscan(nums, sub::num_num));
                return Ok(());
            }
            Some((Primitive::Div, false)) => {
                env.push(fast_invscan(nums, div::num_num));
                return Ok(());
            }
            _ => xs = Value::Num(nums),
        },
        Value::Byte(bytes) => match f.node.as_flipped_primitive() {
            Some((Primitive::Sub, false)) => {
                env.push(fast_invscan(bytes.convert(), sub::num_num));
                return Ok(());
            }
            Some((Primitive::Div, false)) => {
                env.push(fast_invscan(bytes.convert(), div::num_num));
                return Ok(());
            }
            _ => xs = Value::Byte(bytes),
        },
        val => xs = val,
    }

    let mut unscanned = Vec::with_capacity(xs.row_count());
    let mut rows = xs.into_rows();
    let mut curr = rows.next().unwrap();
    unscanned.push(curr.clone());
    env.without_fill(|env| -> UiuaResult {
        for row in rows {
            env.push(row.clone());
            env.push(curr);
            env.exec(f.clone())?;
            unscanned.push(env.pop("unscanned function result")?);
            curr = row;
        }
        Ok(())
    })?;
    env.push(Value::from_row_values(unscanned, env)?);
    Ok(())
}

fn fast_invscan<T>(mut arr: Array<T>, f: impl Fn(T, T) -> T) -> Array<T>
where
    T: ArrayValue + Copy,
{
    match arr.shape.len() {
        0 => unreachable!("fast_invscan called on unit array, should have been guarded against"),
        1 => {
            if arr.row_count() == 0 {
                return arr;
            }
            let mut acc = arr.data[0];
            for val in arr.data.as_mut_slice()[1..].iter_mut() {
                let temp = *val;
                *val = f(acc, *val);
                acc = temp;
            }
            arr
        }
        _ => {
            if arr.row_count() == 0 {
                return arr;
            }
            let row_len: usize = arr.row_len();
            let (acc, rest) = arr.data.as_mut_slice().split_at_mut(row_len);
            let mut acc = acc.to_vec();
            let mut temp = acc.clone();
            for row_slice in rest.chunks_exact_mut(row_len) {
                temp.copy_from_slice(row_slice);
                for (a, b) in acc.iter_mut().zip(row_slice) {
                    *b = f(*a, *b);
                }
                acc.copy_from_slice(&temp);
            }
            arr
        }
    }
}

pub fn fold(ops: Ops, env: &mut Uiua) -> UiuaResult {
    crate::profile_function!();
    let [f] = get_ops(ops, env)?;
    let sig = f.sig;
    let (iterable_count, acc_count, collect_count) = if sig.args > sig.outputs {
        (sig.args - sig.outputs, sig.outputs, 0)
    } else {
        let iter = sig.args.min(1);
        let acc = sig.args.saturating_sub(iter);
        let collect = sig.outputs - acc;
        (iter, acc, collect)
    };
    let mut arrays = Vec::with_capacity(iterable_count);
    for i in 0..iterable_count {
        let mut val = env.pop(("iterated array", i + 1))?;
        arrays.push(if val.row_count() == 1 {
            val.undo_fix();
            Err(val)
        } else {
            Ok(val.into_rows())
        });
    }
    if env.stack_height() < acc_count {
        for i in 0..acc_count {
            env.pop(("accumulator", i + 1))?;
        }
    }
    for i in 0..iterable_count {
        for j in i + 1..iterable_count {
            if let (Ok(a), Ok(b)) = (&arrays[i], &arrays[j]) {
                if a.len() != b.len() {
                    return Err(env.error(format!(
                        "Cannot {} arrays of different lengths: {} and {}",
                        Primitive::Fold.format(),
                        a.len(),
                        b.len()
                    )));
                }
            }
        }
    }
    let mut row_count = arrays
        .iter()
        .filter_map(|arr| arr.as_ref().ok())
        .map(|arr| arr.len())
        .max()
        .unwrap_or(0);
    if row_count == 0 && arrays.iter().all(Result::is_err) {
        row_count = 1;
    }
    let mut collect = multi_output(collect_count, Vec::with_capacity(row_count));
    for _ in 0..row_count {
        for array in arrays.iter_mut().rev() {
            env.push(match array {
                Ok(arr) => arr.next().unwrap(),
                Err(arr) => arr.clone(),
            });
        }
        env.exec(f.clone())?;
        for collected in &mut collect {
            collected.push(env.remove_nth_back(acc_count)?);
        }
    }
    let accs = env.pop_n(acc_count)?;
    for collected in collect.into_iter().rev() {
        let val = Value::from_row_values(collected, env)?;
        env.push(val);
    }
    for acc in accs {
        env.push(acc);
    }
    Ok(())
}

pub fn adjacent(ops: Ops, env: &mut Uiua) -> UiuaResult {
    let [f] = get_ops(ops, env)?;
    let n_arr = env.pop(1)?;
    let xs = env.pop(2)?;
    if n_arr.rank() != 0 {
        return adjacent_fallback(f, n_arr, xs, env);
    }
    let n = n_arr.as_int(env, "Window size must be an integer or list of integers")?;
    if n == 0 {
        return Err(env.error("Window size cannot be zero"));
    }
    let n = if n > 0 {
        n.unsigned_abs()
    } else {
        let count = n.unsigned_abs();
        if count <= xs.row_count() {
            xs.row_count() + 1 - count
        } else {
            return adjacent_fallback(f, n_arr, xs, env);
        }
    };
    match (f.node.as_flipped_primitive(), xs) {
        (Some((prim, flipped)), Value::Num(nums)) => env.push(match prim {
            Primitive::Add => fast_adjacent(nums, n, env, add::num_num),
            Primitive::Sub if flipped => fast_adjacent(nums, n, env, flip(sub::num_num)),
            Primitive::Sub => fast_adjacent(nums, n, env, sub::num_num),
            Primitive::Mul => fast_adjacent(nums, n, env, mul::num_num),
            Primitive::Div if flipped => fast_adjacent(nums, n, env, flip(div::num_num)),
            Primitive::Div => fast_adjacent(nums, n, env, div::num_num),
            Primitive::Modulus if flipped => fast_adjacent(nums, n, env, flip(modulus::num_num)),
            Primitive::Modulus => fast_adjacent(nums, n, env, modulus::num_num),
            Primitive::Atan if flipped => fast_adjacent(nums, n, env, flip(atan2::num_num)),
            Primitive::Atan => fast_adjacent(nums, n, env, atan2::num_num),
            Primitive::Max => fast_adjacent(nums, n, env, max::num_num),
            Primitive::Min => fast_adjacent(nums, n, env, min::num_num),
            _ => return generic_adjacent(f, Value::Num(nums), n, env),
        }?),
        (Some((prim, flipped)), Value::Byte(bytes)) => env.push::<Value>(match prim {
            Primitive::Add => fast_adjacent(bytes.convert(), n, env, add::num_num)?.into(),
            Primitive::Sub if flipped => {
                fast_adjacent(bytes.convert(), n, env, flip(sub::num_num))?.into()
            }
            Primitive::Sub => fast_adjacent(bytes.convert(), n, env, sub::num_num)?.into(),
            Primitive::Mul => fast_adjacent(bytes.convert(), n, env, mul::num_num)?.into(),
            Primitive::Div if flipped => {
                fast_adjacent(bytes.convert(), n, env, flip(div::num_num))?.into()
            }
            Primitive::Div => fast_adjacent(bytes.convert(), n, env, div::num_num)?.into(),
            Primitive::Modulus if flipped => {
                fast_adjacent(bytes.convert(), n, env, flip(modulus::num_num))?.into()
            }
            Primitive::Modulus => fast_adjacent(bytes.convert(), n, env, modulus::num_num)?.into(),
            Primitive::Atan if flipped => {
                fast_adjacent(bytes.convert(), n, env, flip(atan2::num_num))?.into()
            }
            Primitive::Atan => fast_adjacent(bytes.convert(), n, env, atan2::num_num)?.into(),
            Primitive::Max => fast_adjacent(bytes, n, env, max::byte_byte)?.into(),
            Primitive::Min => fast_adjacent(bytes, n, env, min::byte_byte)?.into(),
            _ => return generic_adjacent(f, Value::Byte(bytes), n, env),
        }),
        (_, xs) => generic_adjacent(f, xs, n, env)?,
    }
    Ok(())
}

fn adjacent_fallback(f: SigNode, n: Value, xs: Value, env: &mut Uiua) -> UiuaResult {
    let windows = n.windows(xs, env)?;
    let mut new_rows = Vec::with_capacity(windows.row_count());
    for window in windows.into_rows() {
        env.push(window);
        reduce_impl(f.clone(), 0, env)?;
        new_rows.push(env.pop("adjacent function result")?);
    }
    env.push(Value::from_row_values(new_rows, env)?);
    Ok(())
}

fn fast_adjacent<T>(
    mut arr: Array<T>,
    n: usize,
    env: &Uiua,
    f: impl Fn(T, T) -> T,
) -> UiuaResult<Array<T>>
where
    T: Copy,
{
    match arr.rank() {
        0 => Err(env.error("Cannot get adjacency of scalar")),
        1 => {
            if arr.row_count() < n {
                return Ok(Array::new([0], EcoVec::new()));
            }
            let data = arr.data.as_mut_slice();
            for i in 0..data.len() - (n - 1) {
                let start = i;
                for j in 1..n {
                    data[start] = f(data[start], data[start + j]);
                }
            }
            arr.data.truncate(arr.data.len() - (n - 1));
            arr.shape[0] -= n - 1;
            arr.validate_shape();
            Ok(arr)
        }
        _ => {
            let row_len = arr.row_len();
            let row_count = arr.row_count();
            if row_count < n {
                let mut shape = arr.shape;
                shape[0] = 0;
                return Ok(Array::new(shape, EcoVec::new()));
            }
            let data = arr.data.as_mut_slice();
            for i in 0..row_count - (n - 1) {
                let start = i * row_len;
                for j in 1..n {
                    let next = (i + j) * row_len;
                    for k in 0..row_len {
                        data[start + k] = f(data[start + k], data[next + k]);
                    }
                }
            }
            arr.data.truncate(arr.data.len() - (n - 1) * row_len);
            arr.shape[0] -= n - 1;
            arr.validate_shape();
            Ok(arr)
        }
    }
}

fn generic_adjacent(f: SigNode, xs: Value, n: usize, env: &mut Uiua) -> UiuaResult {
    let sig = f.sig;
    if sig != (2, 1) {
        return Err(env.error(format!(
            "Adjacent's function's signature must be {}, but it is {}",
            Signature::new(2, 1),
            sig
        )));
    }
    if xs.row_count() < n {
        env.push(xs.first_dim_zero());
        return Ok(());
    }
    let win_count = xs.row_count() - (n - 1);
    let mut rows = xs.into_rows();
    let mut window = VecDeque::with_capacity(n);
    let mut new_rows = Vec::with_capacity(win_count);
    window.extend(rows.by_ref().take(n));
    for _ in 0..win_count {
        let mut acc = window.pop_front().unwrap();
        for row in &window {
            env.push(row.clone());
            env.push(acc);
            env.exec(f.clone())?;
            acc = env.pop("adjacent function result")?;
        }
        new_rows.push(acc);
        window.extend(rows.next());
    }
    env.push(Value::from_row_values(new_rows, env)?);
    Ok(())
}
