//! Algorithms for looping modifiers

use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    mem::size_of,
    ptr,
};

use ecow::{eco_vec, EcoVec};

use crate::{
    algorithm::{fixed_rows, FixedRowsData},
    array::{Array, ArrayValue},
    cowslice::CowSlice,
    val_as_arr,
    value::Value,
    Boxed, Function, Instr, Primitive, Shape, Signature, TempStack, Uiua, UiuaResult,
};

use super::{multi_output, validate_size_impl};

pub fn flip<A, B, C>(f: impl Fn(A, B) -> C + Copy) -> impl Fn(B, A) -> C + Copy {
    move |b, a| f(a, b)
}

pub fn repeat(with_inverse: bool, env: &mut Uiua) -> UiuaResult {
    crate::profile_function!();
    let f = env.pop_function()?;
    let inv = with_inverse.then(|| env.pop_function()).transpose()?;
    let n = env.pop("repetition count")?;
    fn rep_count(value: Value, env: &Uiua) -> UiuaResult<Array<f64>> {
        Ok(match value {
            Value::Num(n) => n,
            Value::Byte(n) => n.convert(),
            val => {
                return Err(env.error(format!(
                    "Repetitions must be a scalar or list of \
                    natural numbers or infinity, \
                    but it is {}",
                    val.type_name_plural()
                )))
            }
        })
    }
    if n.rank() == 0 {
        // Scalar repeat
        let n = rep_count(n, env)?;
        repeat_impl(f, inv, n.data[0], env)
    } else {
        // Array
        let sig = f.signature();
        if sig.args != sig.outputs {
            return Err(env.error(format!(
                "{} with a non-scalar repetition count \
                must use a function with the same number \
                of arguments and outputs, but its signature \
                is {sig}",
                Primitive::Repeat.format()
            )));
        }
        // Collect arguments
        let mut args = Vec::with_capacity(sig.args + 1);
        let mut new_shape = n.shape().clone();
        args.push(n);
        for i in 0..sig.args {
            let arg = env.pop(i + 1)?;
            for (a, &b) in new_shape.iter_mut().zip(arg.shape()) {
                *a = (*a).max(b);
            }
            args.push(arg);
        }
        args[1..].reverse();
        let FixedRowsData {
            mut rows,
            row_count,
            is_empty,
            ..
        } = fixed_rows(Primitive::Repeat.format(), sig.outputs, args, env)?;

        // Switch with each selector element
        let mut outputs = multi_output(sig.outputs, Vec::new());
        let mut rows_to_sel = Vec::with_capacity(sig.args);
        for _ in 0..row_count {
            let n = rep_count(
                match &mut rows[0] {
                    Ok(n) => n.next().unwrap(),
                    Err(n) => n.clone(),
                },
                env,
            )?;
            // println!("ns: {} {:?}", n.shape, n.data);
            rows_to_sel.clear();
            for row in rows[1..].iter_mut() {
                let row = match row {
                    Ok(row) => row.next().unwrap(),
                    Err(row) => row.clone(),
                };
                // println!("row: {:?}", row);
                if n.rank() > row.rank() || is_empty {
                    rows_to_sel.push(Err(row));
                } else {
                    let row_shape = row.shape()[n.rank()..].into();
                    rows_to_sel.push(Ok(row.into_row_shaped_slices(row_shape)));
                }
            }
            for sel_row_slice in n.row_slices() {
                for &elem in sel_row_slice {
                    // println!("  elem: {}", elem);
                    for row in &mut rows_to_sel {
                        let row = match row {
                            Ok(row) => row.next().unwrap(),
                            Err(row) => row.clone(),
                        };
                        // println!("  row: {:?}", row);
                        env.push(row);
                    }
                    repeat_impl(f.clone(), inv.clone(), elem, env)?;
                    for i in 0..sig.outputs {
                        let res = env.pop("repeat output")?;
                        // println!("    res: {:?}", res);
                        outputs[i].push(res);
                    }
                }
            }
        }
        // Collect output
        for output in outputs.into_iter().rev() {
            let mut new_value = Value::from_row_values(output, env)?;
            let mut new_shape = new_shape.clone();
            new_shape.extend_from_slice(&new_value.shape()[1..]);
            *new_value.shape_mut() = new_shape;
            new_value.validate_shape();
            if is_empty {
                new_value.pop_row();
            }
            env.push(new_value);
        }
        Ok(())
    }
}

fn repeat_impl(f: Function, inv: Option<Function>, n: f64, env: &mut Uiua) -> UiuaResult {
    let sig = f.signature();
    let (f, n) = if n >= 0.0 {
        (f, n)
    } else {
        let f = inv.ok_or_else(|| env.error("No inverse found"))?;
        (f, -n)
    };
    if n.is_infinite() {
        // Converging repeat
        if sig.args == 0 {
            return Err(env.error(format!(
                "Converging {}'s function must have at least 1 argument",
                Primitive::Repeat.format()
            )));
        }
        if !env.rt.array_stack.is_empty() && sig.args > sig.outputs {
            return Err(env.error(format!(
                "Converging {}'s function must have a net positive stack \
                change inside an array, but its signature is {sig}",
                Primitive::Repeat.format()
            )));
        }
        if env.rt.array_stack.is_empty() && sig.args != sig.outputs {
            return Err(env.error(format!(
                "Converging {}'s function must have a net stack change of 0 \
                outside an array, but its signature is {sig}",
                Primitive::Repeat.format()
            )));
        }
        let mut prev = env.pop(1)?;
        env.push(prev.clone());
        loop {
            env.call(f.clone())?;
            let next = env.pop("converging function result")?;
            let converged = next == prev;
            if converged {
                env.push(next);
                break;
            } else {
                env.push(next.clone());
                prev = next;
            }
        }
    } else {
        // Normal repeat
        if n.fract() != 0.0 {
            return Err(env.error("Repetitions must be an integer or infinity"));
        }
        let n = n as usize;
        if sig.outputs > sig.args {
            let delta = sig.outputs - sig.args;
            if validate_size_impl(size_of::<Value>(), [n, delta]).is_err() {
                return Err(env.error(format!(
                    "{} would create too many values on the stack",
                    Primitive::Repeat.format()
                )));
            }
        }
        for _ in 0..n {
            env.call(f.clone())?;
        }
    }
    Ok(())
}

pub fn do_(env: &mut Uiua) -> UiuaResult {
    crate::profile_function!();
    let body = env.pop_function()?;
    let cond = env.pop_function()?;
    let body_sig = body.signature();
    let cond_sig = cond.signature();
    if cond_sig.outputs < 1 {
        return Err(env.error(format!(
            "Do's condition function must return at least 1 value, \
            but its signature is {cond_sig}"
        )));
    }
    let copy_count = cond_sig.args.saturating_sub(cond_sig.outputs - 1);
    let cond_sub_sig = Signature::new(cond_sig.args, cond_sig.outputs + copy_count - 1);
    let comp_sig = body_sig.compose(cond_sub_sig);
    match comp_sig.args.cmp(&comp_sig.outputs) {
        Ordering::Less if env.rt.array_stack.is_empty() => {
            return Err(env.error(format!(
                "Do's functions cannot have a positive net stack \
                change outside an array, but the composed signature of \
                {body_sig} and {cond_sig}, minus the condition, is {comp_sig}"
            )))
        }
        Ordering::Greater => {
            return Err(env.error(format!(
                "Do's functions cannot have a negative net stack \
                change, but the composed signature of {body_sig} and \
                {cond_sig}, minus the condition, is {comp_sig}"
            )))
        }
        _ => {}
    }
    loop {
        // Make sure there are enough values
        if env.stack().len() < copy_count {
            // Pop until it fails
            for i in 0..copy_count {
                env.pop(i + 1)?;
            }
        }
        // Copy necessary condition args
        env.dup_n(copy_count)?;
        // Call condition
        env.call(cond.clone())?;
        // Break if condition is false
        let cond = (env.pop("do condition")?).as_bool(env, "Do condition must be a boolean")?;
        if !cond {
            break;
        }
        // Call body
        env.call(body.clone())?;
    }
    Ok(())
}

pub fn partition(env: &mut Uiua) -> UiuaResult {
    crate::profile_function!();
    collapse_groups(
        Primitive::Partition,
        Value::partition_groups,
        |val, markers, _| Ok(val.partition_firsts(markers)),
        |val, markers, _| Ok(val.partition_lasts(markers)),
        partition_lens,
        Value::first_partition,
        Value::last_partition,
        "⊜ partition indices array must be a list of integers",
        env,
    )
}

impl Value {
    fn partition_groups(self, markers: Array<isize>) -> Box<dyn ExactSizeIterator<Item = Self>> {
        val_as_arr!(self, |arr| arr.partition_groups(markers))
    }
    fn partition_firsts(self, markers: &[isize]) -> Self {
        val_as_arr!(self, |arr| arr.partition_firsts(markers).into())
    }
    fn partition_lasts(self, markers: &[isize]) -> Self {
        val_as_arr!(self, |arr| arr.partition_lasts(markers).into())
    }
    fn first_partition(self, markers: &[isize]) -> Option<Self> {
        Some(val_as_arr!(self, |arr| arr
            .first_partition(markers)?
            .into()))
    }
    fn last_partition(self, markers: &[isize]) -> Option<Self> {
        Some(val_as_arr!(self, |arr| arr.last_partition(markers)?.into()))
    }
}

struct PartitionIter<T> {
    len: usize,
    curr: usize,
    markers: CowSlice<isize>,
    source: Array<T>,
}

impl<T> Iterator for PartitionIter<T>
where
    T: Clone,
    Array<T>: Into<Value>,
{
    type Item = Value;
    fn next(&mut self) -> Option<Self::Item> {
        let row_len = self.source.row_len();
        while self.curr < self.markers.len() {
            let marker = self.markers[self.curr];
            if marker <= 0 {
                self.curr += 1;
            } else {
                let start = self.curr;
                while self.curr < self.markers.len() && self.markers[self.curr] == marker {
                    self.curr += 1;
                }
                let end = self.curr;
                let data = self.source.data.slice(start * row_len..end * row_len);
                let mut shape = self.source.shape.clone();
                shape[0] = end - start;
                return Some(Array::new(shape, data).into());
            }
        }
        None
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len - self.curr;
        (len, Some(len))
    }
}

impl<T> ExactSizeIterator for PartitionIter<T>
where
    T: Clone,
    Array<T>: Into<Value>,
{
}

impl<T: ArrayValue> Array<T>
where
    Array<T>: Into<Value>,
{
    fn partition_groups(self, markers: Array<isize>) -> Box<dyn ExactSizeIterator<Item = Value>> {
        let mut groups = Vec::new();
        if markers.rank() == 1 {
            let mut count = 0;
            let mut last_marker = isize::MAX;
            for &marker in &markers.data {
                if marker > 0 && marker != last_marker {
                    count += 1;
                }
                last_marker = marker;
            }
            Box::new(PartitionIter {
                len: count,
                curr: 0,
                markers: markers.data,
                source: self,
            })
        } else {
            let row_shape: Shape = self.shape()[markers.rank()..].into();
            let indices = multi_partition_indices(markers);
            for (_, indices) in indices {
                let mut group = Vec::with_capacity(indices.len());
                for index in indices {
                    group.push(self.row_shaped_slice(index, row_shape.clone()));
                }
                groups.push(group);
            }
            Box::new(
                groups
                    .into_iter()
                    .map(Array::from_row_arrays_infallible)
                    .map(Into::into),
            )
        }
    }
    fn partition_firsts(mut self, markers: &[isize]) -> Self {
        if self.shape.len() == 0 {
            self.shape.insert(0, 0);
        } else {
            self.shape[0] = 0;
        }
        let row_len = self.row_len();
        let data = self.data.as_mut_slice();
        let mut last_marker = isize::MAX;
        for (i, &marker) in markers.iter().enumerate() {
            if marker > 0 && marker != last_marker {
                let dest_start = self.shape[0] * row_len;
                let src_start = i * row_len;
                if dest_start != src_start {
                    unsafe {
                        ptr::swap_nonoverlapping(
                            data.as_mut_ptr().add(dest_start),
                            data.as_mut_ptr().add(src_start),
                            row_len,
                        );
                    }
                }
                self.shape[0] += 1;
            }
            last_marker = marker;
        }
        self.data.truncate(self.shape[0] * row_len);
        self.validate_shape();
        self
    }
    fn partition_lasts(mut self, markers: &[isize]) -> Self {
        let row_count = self.row_count();
        if self.shape.len() == 0 {
            self.shape.insert(0, 0);
        } else {
            self.shape[0] = 0;
        }
        let row_len = self.row_len();
        let data = self.data.as_mut_slice();
        let mut last_marker = isize::MAX;
        for (i, &marker) in markers.iter().enumerate().rev() {
            if marker > 0 && marker != last_marker {
                self.shape[0] += 1;
                let dest_start = (row_count - self.shape[0]) * row_len;
                let src_start = i * row_len;
                if dest_start != src_start {
                    unsafe {
                        ptr::swap_nonoverlapping(
                            data.as_mut_ptr().add(dest_start),
                            data.as_mut_ptr().add(src_start),
                            row_len,
                        );
                    }
                }
            }
            last_marker = marker;
        }
        data.rotate_right(self.shape[0] * row_len);
        self.data.truncate(self.shape[0] * row_len);
        self.validate_shape();
        self
    }
    fn first_partition(mut self, markers: &[isize]) -> Option<Self> {
        let first_marker = markers.first().copied()?;
        let count = markers
            .iter()
            .take_while(|&&marker| marker == first_marker)
            .count();
        let row_len = self.row_len();
        self.data.truncate(count * row_len);
        self.shape[0] = count;
        self.validate_shape();
        Some(self)
    }
    fn last_partition(mut self, markers: &[isize]) -> Option<Self> {
        let last_marker = markers.last().copied()?;
        let count = (markers.iter().rev())
            .take_while(|&&marker| marker == last_marker)
            .count();
        let row_len = self.row_len();
        self.data.as_mut_slice().rotate_right(count * row_len);
        self.data.truncate(count * row_len);
        self.shape[0] = count;
        self.validate_shape();
        Some(self)
    }
}

fn partition_lens(markers: &[isize]) -> Array<f64> {
    let mut lens = EcoVec::new();
    let mut prev = isize::MAX;
    let mut len = 0;
    for &marker in markers {
        if marker > 0 {
            if marker == prev {
                len += 1;
            } else {
                if len > 0 {
                    lens.push(len as f64);
                }
                len = 1;
            }
        } else {
            if len > 0 {
                lens.push(len as f64);
            }
            len = 0;
        }
        prev = marker;
    }
    if len > 0 {
        lens.push(len as f64);
    }
    lens.into()
}

fn multi_partition_indices(markers: Array<isize>) -> Vec<(isize, Vec<usize>)> {
    if markers.element_count() == 0 {
        return Vec::new();
    }
    let mut groups: Vec<(isize, Vec<Vec<usize>>)> = Vec::new();
    let mut curr = vec![0; markers.rank()];
    for &marker in &markers.data {
        if marker >= 1 {
            let mut adjacent_groups = Vec::new();
            // Find adjacent groups with the same value
            for (g, (val, group)) in groups.iter().enumerate() {
                if *val != marker {
                    continue;
                }
                if group.iter().any(|idx| {
                    idx.iter()
                        .zip(&curr)
                        .map(|(a, b)| a.abs_diff(*b))
                        .sum::<usize>()
                        == 1
                }) {
                    adjacent_groups.push(g);
                }
            }
            // Add the current index to the adjacent group, possibly merging groups
            match adjacent_groups.len() {
                0 => {
                    groups.push((marker, vec![curr.clone()]));
                }
                1 => groups[adjacent_groups[0]].1.push(curr.clone()),
                _ => {
                    let mut new_group = Vec::new();
                    for (i, g) in adjacent_groups.into_iter().enumerate() {
                        new_group.extend(groups.remove(g - i).1);
                    }
                    new_group.push(curr.clone());
                    groups.push((marker, new_group));
                }
            }
        }
        // Increment the current index
        for (i, c) in curr.iter_mut().enumerate().rev() {
            if *c < markers.shape()[i] - 1 {
                *c += 1;
                break;
            }
            *c = 0;
        }
    }
    let mut shape_muls: Vec<usize> = (markers.shape().iter().rev())
        .scan(1, |mul, &dim| {
            let prev = *mul;
            *mul *= dim;
            Some(prev)
        })
        .collect();
    shape_muls.reverse();
    groups
        .into_iter()
        .map(|(marker, group)| {
            (
                marker,
                (group.into_iter())
                    .map(|index| index.iter().zip(&shape_muls).map(|(i, m)| i * m).sum())
                    .collect(),
            )
        })
        .collect()
}

pub fn undo_partition_part1(env: &mut Uiua) -> UiuaResult {
    crate::profile_function!();
    let f = env.pop_function()?;
    let sig = f.signature();
    if sig != (1, 1) {
        return Err(env.error(format!(
            "Cannot undo {} on function with signature {sig}",
            Primitive::Partition.format()
        )));
    }
    let partitioned = env.pop(1)?;
    // Untransform rows
    let mut untransformed = Vec::with_capacity(partitioned.row_count());
    for row in partitioned.into_rows().rev() {
        env.push(row);
        env.call(f.clone())?;
        untransformed.push(Boxed(env.pop("unpartitioned row")?));
    }
    untransformed.reverse();
    env.push(Array::from_iter(untransformed));
    Ok(())
}

#[allow(clippy::unit_arg)]
pub fn undo_partition_part2(env: &mut Uiua) -> UiuaResult {
    let untransformed = env.pop(1)?;
    let markers = env
        .pop(2)?
        .as_integer_array(env, "⊜ partition markers must be an array of integers")?;
    let mut original = env.pop(3)?;
    if markers.rank() == 1 {
        // Count partition markers
        let mut marker_partitions: Vec<(isize, usize)> = Vec::new();
        let mut markers = markers.data.into_iter();
        if let Some(mut prev) = markers.next() {
            marker_partitions.push((prev, 1));
            for marker in markers {
                if marker == prev {
                    marker_partitions.last_mut().unwrap().1 += 1;
                } else {
                    marker_partitions.push((marker, 1));
                }
                prev = marker;
            }
        }
        let positive_partitions = marker_partitions.iter().filter(|(m, _)| *m > 0).count();
        if positive_partitions != untransformed.row_count() {
            return Err(env.error(format!(
                "Cannot undo {} because the partitioned array \
                originally had {} rows, but now it has {}",
                Primitive::Partition.format(),
                positive_partitions,
                untransformed.row_count()
            )));
        }

        // Unpartition
        let mut untransformed_rows = untransformed.into_rows().map(Value::unboxed);
        let mut unpartitioned = Vec::with_capacity(marker_partitions.len() * original.row_len());
        let mut original_offset = 0;
        for (marker, part_len) in marker_partitions {
            if marker > 0 {
                unpartitioned.extend(untransformed_rows.next().unwrap().into_rows());
            } else {
                unpartitioned
                    .extend((original_offset..original_offset + part_len).map(|i| original.row(i)));
            }
            original_offset += part_len;
        }
        env.push(Value::from_row_values(unpartitioned, env)?);
    } else {
        let row_shape: Shape = original.shape()[markers.rank()..].into();
        let indices = multi_partition_indices(markers);
        let row_elem_count: usize = row_shape.iter().product();
        let untransformed_rows = untransformed.into_rows().map(Value::unboxed);
        for ((_, indices), untransformed_row) in indices.into_iter().zip(untransformed_rows) {
            if indices.len() != untransformed_row.row_count() {
                return Err(env.error(format!(
                    "Cannot undo {} because a partitioned array's \
                    row's length changed from {} to {}",
                    Primitive::Partition.format(),
                    indices.len(),
                    untransformed_row.row_count(),
                )));
            }
            for (index, row) in indices.into_iter().zip(untransformed_row.into_rows()) {
                let start = index * row_elem_count;
                original.generic_bin_mut(
                    row,
                    |a, b| Ok(update_array_at(a, start, b.data.as_slice())),
                    |a, b| Ok(update_array_at(a, start, b.data.as_slice())),
                    |a, b| Ok(update_array_at(a, start, b.data.as_slice())),
                    |a, b| Ok(update_array_at(a, start, b.data.as_slice())),
                    |a, b| Ok(update_array_at(a, start, b.data.as_slice())),
                    |a, b| {
                        env.error(format!(
                            "Cannot unpartition {} array into {} array",
                            b.type_name(),
                            a.type_name()
                        ))
                    },
                )?;
            }
        }
        env.push(original);
    }
    Ok(())
}

fn update_array_at<T: Clone>(arr: &mut Array<T>, start: usize, new: &[T]) {
    let end = start + new.len();
    arr.data.as_mut_slice()[start..end].clone_from_slice(new);
}

pub fn group(env: &mut Uiua) -> UiuaResult {
    crate::profile_function!();
    collapse_groups(
        Primitive::Group,
        Value::group_groups,
        Value::group_firsts,
        Value::group_lasts,
        |indices| {
            let buckets = (indices.iter().max().copied().unwrap_or(-1) + 1).max(0) as usize;
            let mut len_counts = HashMap::with_capacity(buckets);
            for &index in indices {
                if index >= 0 {
                    *len_counts.entry(index.unsigned_abs()).or_insert(0) += 1;
                }
            }
            let mut lens: EcoVec<f64> = eco_vec![0.0; buckets];
            let slice = lens.make_mut();
            for (index, len) in len_counts {
                slice[index] = len as f64;
            }
            lens.into()
        },
        Value::first_group,
        Value::last_group,
        "⊕ group indices array must be an array of integers",
        env,
    )
}

impl Value {
    fn group_groups(self, indices: Array<isize>) -> Vec<Self> {
        val_as_arr!(self, |arr| arr
            .group_groups(indices)
            .map(Into::into)
            .collect())
    }
    fn group_firsts(self, indices: &[isize], env: &Uiua) -> UiuaResult<Self> {
        Ok(val_as_arr!(self, |arr| arr
            .group_firsts(indices, env)?
            .into()))
    }
    fn group_lasts(self, indices: &[isize], env: &Uiua) -> UiuaResult<Self> {
        Ok(val_as_arr!(self, |arr| arr
            .group_lasts(indices, env)?
            .into()))
    }
    fn first_group(self, indices: &[isize]) -> Option<Self> {
        Some(val_as_arr!(self, |arr| arr.first_group(indices)?.into()))
    }
    fn last_group(self, indices: &[isize]) -> Option<Self> {
        Some(val_as_arr!(self, |arr| arr.last_group(indices)?.into()))
    }
}

impl<T: ArrayValue> Array<T> {
    fn group_groups(self, indices: Array<isize>) -> impl Iterator<Item = Self> {
        let Some(&max_index) = indices.data.iter().max() else {
            return Vec::<Vec<Self>>::new()
                .into_iter()
                .map(Array::from_row_arrays_infallible);
        };
        let buckets = (max_index.max(-1) + 1).max(0) as usize;
        let mut groups: Vec<Vec<Self>> = vec![Vec::new(); buckets];
        let row_shape = self.shape()[indices.rank()..].into();
        for (g, r) in (indices.data.into_iter()).zip(self.into_row_shaped_slices(row_shape)) {
            if g >= 0 && g < buckets as isize {
                groups[g as usize].push(r);
            }
        }
        groups.into_iter().map(Array::from_row_arrays_infallible)
    }
    fn group_firsts(self, indices: &[isize], env: &Uiua) -> UiuaResult<Self> {
        let buckets = (indices.iter().copied().max().unwrap_or(-1) + 1).max(0) as usize;
        if self.row_count() < buckets {
            return Err(env.error("Cannot take first because a group was empty"));
        }
        let row_len = self.row_len();
        let mut encountered = HashSet::new();
        let mut data = self.data.clone();
        data.truncate(buckets * row_len);
        let data_slice = data.as_mut_slice();
        let mut shape = self.shape.clone();
        if shape.len() == 0 {
            shape.insert(0, buckets);
        } else {
            shape[0] = buckets;
        }
        for (&index, row) in indices.iter().zip(self.row_slices()) {
            if index >= 0 && encountered.insert(index) {
                let start = index.unsigned_abs() * row_len;
                let end = start + row_len;
                data_slice[start..end].clone_from_slice(row);
            }
        }
        if encountered.len() != buckets {
            return Err(env.error("Cannot take first because a group was empty"));
        }
        Ok(Array::new(shape, data))
    }
    fn group_lasts(self, indices: &[isize], env: &Uiua) -> UiuaResult<Self> {
        let buckets = (indices.iter().copied().max().unwrap_or(-1) + 1).max(0) as usize;
        if self.row_count() < buckets {
            return Err(env.error("Cannot take last because a group was empty"));
        }
        let row_len = self.row_len();
        let mut encountered = HashSet::new();
        let mut data = self.data.clone();
        data.truncate(buckets * row_len);
        let data_slice = data.as_mut_slice();
        let mut shape = self.shape.clone();
        if shape.len() == 0 {
            shape.insert(0, buckets);
        } else {
            shape[0] = buckets;
        }
        for (&index, row) in indices.iter().zip(self.row_slices()).rev() {
            if index >= 0 && encountered.insert(index) {
                let start = index.unsigned_abs() * row_len;
                let end = start + row_len;
                data_slice[start..end].clone_from_slice(row);
            }
        }
        if encountered.len() != buckets {
            return Err(env.error("Cannot take last because a group was empty"));
        }
        Ok(Array::new(shape, data))
    }
    fn first_group(mut self, indices: &[isize]) -> Option<Self> {
        if indices.is_empty() {
            return None;
        }
        let mut dest = 0;
        let row_len = self.row_len();
        let slice = self.data.as_mut_slice();
        for (src, &index) in indices.iter().enumerate() {
            if index == 0 {
                if src != dest {
                    let src_start = src * row_len;
                    let dest_start = dest * row_len;
                    for i in 0..row_len {
                        slice[dest_start + i] = slice[src_start + i].clone();
                    }
                }
                dest += 1;
            }
        }
        self.data.truncate(dest * row_len);
        self.shape[0] = dest;
        self.validate_shape();
        Some(self)
    }
    fn last_group(mut self, indices: &[isize]) -> Option<Self> {
        let last_group = indices.iter().max().copied()?;
        let mut dest = 0;
        let row_len = self.row_len();
        let slice = self.data.as_mut_slice();
        for (src, &index) in indices.iter().enumerate() {
            if index == last_group {
                if src != dest {
                    let src_start = src * row_len;
                    let dest_start = dest * row_len;
                    for i in 0..row_len {
                        slice[dest_start + i] = slice[src_start + i].clone();
                    }
                }
                dest += 1;
            }
        }
        self.data.truncate(dest * row_len);
        self.shape[0] = dest;
        self.validate_shape();
        Some(self)
    }
}

pub fn undo_group_part1(env: &mut Uiua) -> UiuaResult {
    crate::profile_function!();
    let f = env.pop_function()?;
    let sig = f.signature();
    if sig != (1, 1) {
        return Err(env.error(format!(
            "Cannot undo {} on function with signature {sig}",
            Primitive::Group.format()
        )));
    }
    let grouped = env.pop(1)?;

    // Untransform rows
    let mut ungrouped_rows = Vec::with_capacity(grouped.row_count());
    for mut row in grouped.into_rows().rev() {
        env.push(row);
        env.call(f.clone())?;
        row = env.pop("ungrouped row")?;
        ungrouped_rows.push(Boxed(row));
    }
    ungrouped_rows.reverse();
    env.push(Array::from_iter(ungrouped_rows));
    Ok(())
}

pub fn undo_group_part2(env: &mut Uiua) -> UiuaResult {
    let ungrouped_rows = env.pop(1)?;
    let indices = env
        .pop(2)?
        .as_integer_array(env, "⊕ group indices must be an array of integers")?;
    let original = env.pop(3)?;

    let expected_count = (indices.data.iter().max().copied().unwrap_or(-1) + 1).max(0);
    if ungrouped_rows.row_count() as isize != expected_count {
        return Err(env.error(format!(
            "Cannot undo {} because the grouped array's \
            length changed from {} to {}",
            Primitive::Group.format(),
            expected_count,
            ungrouped_rows.row_count(),
        )));
    }

    // Ungroup
    let mut ungrouped_rows: Vec<_> = ungrouped_rows
        .into_rows()
        .map(|row| row.unboxed().into_rows())
        .collect();
    let mut ungrouped = Vec::with_capacity(indices.element_count() * original.row_len());
    let depth = indices.rank().saturating_sub(1);
    for (i, &index) in indices.data.iter().enumerate() {
        let original_row = original.depth_row(depth, i);
        if index >= 0 {
            ungrouped.push(ungrouped_rows[index as usize].next().ok_or_else(|| {
                env.error("A group's length was modified between grouping and ungrouping")
            })?);
        } else {
            ungrouped.push(original_row);
        }
    }
    if ungrouped_rows.iter_mut().any(|row| row.next().is_some()) {
        return Err(env.error("A group's length was modified between grouping and ungrouping"));
    }
    let mut val = Value::from_row_values(ungrouped, env)?;
    val.shape_mut().remove(0);
    for &dim in indices.shape().iter().rev() {
        val.shape_mut().insert(0, dim);
    }
    val.validate_shape();
    env.push(val);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn collapse_groups<I>(
    prim: Primitive,
    get_groups: impl Fn(Value, Array<isize>) -> I,
    firsts: impl Fn(Value, &[isize], &Uiua) -> UiuaResult<Value>,
    lasts: impl Fn(Value, &[isize], &Uiua) -> UiuaResult<Value>,
    lens: impl Fn(&[isize]) -> Array<f64>,
    first_group: impl Fn(Value, &[isize]) -> Option<Value>,
    last_group: impl Fn(Value, &[isize]) -> Option<Value>,
    indices_error: &'static str,
    env: &mut Uiua,
) -> UiuaResult
where
    I: IntoIterator<Item = Value>,
    I::IntoIter: ExactSizeIterator,
{
    let f = env.pop_function()?;
    let sig = f.signature();
    let indices = env.pop(1)?.as_integer_array(env, indices_error)?;
    let values = env.pop(2)?;

    if !values.shape().starts_with(indices.shape()) {
        return Err(env.error(format!(
            "Cannot {} array of shape {} with indices of shape {}",
            prim.format(),
            values.shape(),
            indices.shape()
        )));
    }

    // Optimizations
    if indices.rank() == 1 {
        use Instr::*;
        use Primitive::*;
        match f.instrs(&env.asm) {
            [Prim(First, _)] => {
                let val = firsts(values, &indices.data, env)?;
                env.push(val);
                return Ok(());
            }
            [Prim(Last, _)] => {
                let val = lasts(values, &indices.data, env)?;
                env.push(val);
                return Ok(());
            }
            [Prim(Len, _)] => {
                if indices.row_count() != values.row_count() {
                    return Err(env.error(format!(
                        "Cannot {} array of shape {} with indices of shape {}",
                        prim.format(),
                        values.shape(),
                        indices.shape()
                    )));
                }
                env.push(lens(&indices.data));
                return Ok(());
            }
            [Prim(Pop, _), Prim(Identity, _)] => {
                let val = last_group(values, &indices.data).ok_or_else(|| {
                    env.error(format!(
                        "Cannot do aggregating {} with no groups",
                        prim.format()
                    ))
                })?;
                env.push(val);
                return Ok(());
            }
            [PushTemp {
                stack: TempStack::Inline,
                count: 1,
                ..
            }, Prim(Pop, _), PopTemp {
                stack: TempStack::Inline,
                count: 1,
                ..
            }] => {
                let val = first_group(values, &indices.data).ok_or_else(|| {
                    env.error(format!(
                        "Cannot do aggregating {} with no groups",
                        prim.format()
                    ))
                })?;
                env.push(val);
                return Ok(());
            }
            _ => {}
        }
    }

    match (sig.args, sig.outputs) {
        (0 | 1, outputs) => {
            let mut empty_shape = values.shape().clone();
            let is_scalar = empty_shape.is_empty();
            *empty_shape.row_count_mut() = 0;
            let groups = get_groups(values, indices).into_iter().map(|mut group| {
                if group.row_count() == 0 {
                    group.shape_mut().clone_from(&empty_shape);
                    group.validate_shape();
                }
                group
            });
            let mut rows = multi_output(outputs, Vec::with_capacity(groups.len()));
            env.without_fill(|env| -> UiuaResult {
                for group in groups {
                    env.push(group);
                    env.call(f.clone())?;
                    for i in 0..outputs {
                        let value = env.pop(|| format!("{}'s function result", prim.format()))?;
                        rows[i].push(value);
                    }
                    if sig.args == 0 {
                        env.pop("excess value")?;
                    }
                }
                Ok(())
            })?;
            for rows in rows.into_iter().rev() {
                let mut val = Value::from_row_values(rows, env)?;
                if is_scalar {
                    val.undo_fix();
                }
                env.push(val);
            }
        }
        (2, 1) => {
            let mut groups = get_groups(values, indices).into_iter();
            let mut acc = match env.value_fill().cloned() {
                Some(acc) => acc,
                None => groups.next().ok_or_else(|| {
                    env.error(format!(
                        "Cannot do aggregating {} with no groups",
                        prim.format()
                    ))
                })?,
            };
            env.without_fill(|env| -> UiuaResult {
                for row in groups {
                    env.push(row);
                    env.push(acc);
                    env.call(f.clone())?;
                    acc = env.pop("reduced function result")?;
                }
                env.push(acc);
                Ok(())
            })?;
        }
        _ => {
            return Err(env.error(format!(
                "Cannot {} with a function with signature {sig}",
                prim.format()
            )))
        }
    }
    Ok(())
}
