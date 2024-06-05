//! Algorithms for pervasive array operations

use std::{
    cmp::Ordering, convert::Infallible, fmt::Display, marker::PhantomData, slice::ChunksExact,
};

use ecow::eco_vec;

use crate::Complex;
use crate::{array::*, Uiua, UiuaError, UiuaResult};

use super::{fill_array_shapes, FillContext};

pub(crate) struct ArrayRef<'a, T> {
    shape: &'a [usize],
    data: &'a [T],
}

impl<'a, T> Clone for ArrayRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for ArrayRef<'a, T> {}

impl<'a, T> ArrayRef<'a, T> {
    pub fn new(shape: &'a [usize], data: &'a [T]) -> Self {
        Self { shape, data }
    }
    fn row_len(&self) -> usize {
        self.shape.iter().skip(1).product()
    }
    fn rows(&self) -> ChunksExact<T> {
        self.data.chunks_exact(self.row_len().max(1))
    }
}

impl<'a, T> From<&'a Array<T>> for ArrayRef<'a, T> {
    fn from(a: &'a Array<T>) -> Self {
        Self::new(a.shape(), a.data.as_slice())
    }
}

pub trait PervasiveFn<A, B> {
    type Output;
    type Error;
    fn call(&self, a: A, b: B, env: &Uiua) -> Result<Self::Output, Self::Error>;
}

#[derive(Clone)]
pub struct InfalliblePervasiveFn<A, B, C, F>(F, PhantomData<(A, B, C)>);

impl<A, B, C, F> InfalliblePervasiveFn<A, B, C, F> {
    pub fn new(f: F) -> Self {
        Self(f, PhantomData)
    }
}

impl<A, B, C, F> PervasiveFn<A, B> for InfalliblePervasiveFn<A, B, C, F>
where
    F: Fn(A, B) -> C,
{
    type Output = C;
    type Error = Infallible;
    fn call(&self, a: A, b: B, _env: &Uiua) -> Result<Self::Output, Self::Error> {
        Ok((self.0)(a, b))
    }
}

#[derive(Clone)]
pub struct FalliblePerasiveFn<A, B, C, F>(F, PhantomData<(A, B, C)>);

impl<A, B, C, F> FalliblePerasiveFn<A, B, C, F> {
    pub fn new(f: F) -> Self {
        Self(f, PhantomData)
    }
}

impl<A, B, C, F> PervasiveFn<A, B> for FalliblePerasiveFn<A, B, C, F>
where
    F: Fn(A, B, &Uiua) -> UiuaResult<C>,
{
    type Output = C;
    type Error = UiuaError;
    fn call(&self, a: A, b: B, env: &Uiua) -> UiuaResult<Self::Output> {
        (self.0)(a, b, env)
    }
}

pub fn bin_pervade<A, B, C, F>(
    mut a: Array<A>,
    mut b: Array<B>,
    a_depth: usize,
    b_depth: usize,
    env: &Uiua,
    f: F,
) -> UiuaResult<Array<C>>
where
    A: ArrayValue,
    B: ArrayValue,
    C: ArrayValue,
    F: PervasiveFn<A, B, Output = C> + Clone,
    F::Error: Into<UiuaError>,
{
    // Fast fixed cases
    if a_depth == 0 && b_depth == 0 && env.scalar_fill::<C>().is_err() {
        // A is fixed
        if a.row_count() == 1 && b.row_count() != 1 {
            let fix_count = a.shape.iter().take_while(|&&d| d == 1).count();
            if b.rank() > fix_count {
                if (a.shape().iter())
                    .zip(b.shape())
                    .skip(fix_count)
                    .any(|(a, b)| a != b)
                {
                    return Err(env.error(format!(
                        "Shapes {} and {} do not match",
                        a.shape(),
                        b.shape()
                    )));
                }
                let mut data = eco_vec![C::default(); b.element_count()];
                let a_row_shape = &a.shape()[fix_count..];
                let b_row_shape = &b.shape()[fix_count..];
                let b_row_len = b_row_shape.iter().product();
                if b_row_len > 0 {
                    for (b, c) in (b.data.chunks_exact(b_row_len))
                        .zip(data.make_mut().chunks_exact_mut(b_row_len))
                    {
                        bin_pervade_recursive(
                            ArrayRef::new(a_row_shape, &a.data),
                            ArrayRef::new(b_row_shape, b),
                            c,
                            env,
                            f.clone(),
                        )
                        .map_err(Into::into)?;
                    }
                }
                return Ok(Array::new(b.shape, data));
            }
        }
        // B is fixed
        if a.row_count() != 1 && b.row_count() == 1 {
            let fix_count = b.shape.iter().take_while(|&&d| d == 1).count();
            if a.rank() > fix_count {
                if (a.shape().iter())
                    .zip(b.shape())
                    .skip(fix_count)
                    .any(|(a, b)| a != b)
                {
                    return Err(env.error(format!(
                        "Shapes {} and {} do not match",
                        a.shape(),
                        b.shape()
                    )));
                }
                let mut data = eco_vec![C::default(); a.element_count()];
                let a_row_shape = &a.shape()[fix_count..];
                let b_row_shape = &b.shape()[fix_count..];
                let a_row_len = a_row_shape.iter().product();
                if a_row_len > 0 {
                    for (a, c) in (a.data.chunks_exact(a_row_len))
                        .zip(data.make_mut().chunks_exact_mut(a_row_len))
                    {
                        bin_pervade_recursive(
                            ArrayRef::new(a_row_shape, a),
                            ArrayRef::new(b_row_shape, &b.data),
                            c,
                            env,
                            f.clone(),
                        )
                        .map_err(Into::into)?;
                    }
                }
                return Ok(Array::new(a.shape, data));
            }
        }
    }
    // Fill
    fill_array_shapes(&mut a, &mut b, a_depth, b_depth, env)?;
    // Pervade
    let shape = a.shape().max(b.shape()).clone();
    let mut data = eco_vec![C::default(); shape.elements()];
    bin_pervade_recursive((&a).into(), (&b).into(), data.make_mut(), env, f).map_err(Into::into)?;
    Ok(Array::new(shape, data))
}

pub fn bin_pervade_recursive<A, B, C, F>(
    a: ArrayRef<A>,
    b: ArrayRef<B>,
    c: &mut [C],
    env: &Uiua,
    f: F,
) -> Result<(), F::Error>
where
    A: ArrayValue,
    B: ArrayValue,
    C: ArrayValue,
    F: PervasiveFn<A, B, Output = C> + Clone,
{
    match (a.shape, b.shape) {
        ([], []) => c[0] = f.call(a.data[0].clone(), b.data[0].clone(), env)?,
        (ash, bsh) if ash.contains(&0) || bsh.contains(&0) => {}
        (ash, bsh) if ash == bsh => {
            for ((a, b), c) in a.data.iter().zip(b.data).zip(c) {
                *c = f.call(a.clone(), b.clone(), env)?;
            }
        }
        ([], bsh) => {
            for (brow, crow) in b.rows().zip(c.chunks_exact_mut(b.row_len())) {
                bin_pervade_recursive(a, ArrayRef::new(&bsh[1..], brow), crow, env, f.clone())?;
            }
        }
        (ash, []) => {
            for (arow, crow) in a.rows().zip(c.chunks_exact_mut(a.row_len())) {
                bin_pervade_recursive(ArrayRef::new(&ash[1..], arow), b, crow, env, f.clone())?;
            }
        }
        (ash, bsh) => {
            for ((arow, brow), crow) in
                (a.rows().zip(b.rows())).zip(c.chunks_exact_mut(a.row_len().max(b.row_len())))
            {
                bin_pervade_recursive(
                    ArrayRef::new(&ash[1..], arow),
                    ArrayRef::new(&bsh[1..], brow),
                    crow,
                    env,
                    f.clone(),
                )?;
            }
        }
    }
    Ok(())
}

pub fn bin_pervade_mut<T>(
    mut a: Array<T>,
    b: &mut Array<T>,
    a_depth: usize,
    b_depth: usize,
    env: &Uiua,
    f: impl Fn(T, T) -> T + Copy,
) -> UiuaResult
where
    T: ArrayValue + Copy,
{
    // Fast case if A is fixed
    if a.row_count() == 1 && b.row_count() != 1 && env.scalar_fill::<T>().is_err() {
        let fix_count = a.shape.iter().take_while(|&&d| d == 1).count();
        if b.rank() > fix_count {
            if (a.shape().iter())
                .zip(b.shape())
                .skip(fix_count)
                .any(|(a, b)| a != b)
            {
                return Err(env.error(format!(
                    "Shapes {} and {} do not match",
                    a.shape(),
                    b.shape()
                )));
            }
            let a_row_shape = &a.shape()[fix_count..];
            let b_row_shape = b.shape()[fix_count..].to_vec();
            let b_row_len = b_row_shape.iter().product();
            if b_row_len > 0 {
                for b in b.data.as_mut_slice().chunks_exact_mut(b_row_len) {
                    bin_pervade_recursive_mut_right(
                        a.data.as_slice(),
                        a_row_shape,
                        b,
                        &b_row_shape,
                        f,
                    );
                }
            }
            return Ok(());
        }
    }
    // Fill
    fill_array_shapes(&mut a, b, a_depth, b_depth, env)?;
    // Pervade
    let ash = a.shape.dims();
    let bsh = b.shape.dims();
    // Try to avoid copying when possible
    if ash == bsh {
        if a.data.is_copy_of(&b.data) {
            drop(a);
            let b_data = b.data.as_mut_slice();
            for b in b_data {
                *b = f(*b, *b);
            }
        } else if a.data.is_unique() {
            let a_data = a.data.as_mut_slice();
            let b_data = b.data.as_slice();
            for (a, b) in a_data.iter_mut().zip(b_data) {
                *a = f(*a, *b);
            }
            *b = a;
        } else {
            let a_data = a.data.as_slice();
            let b_data = b.data.as_mut_slice();
            for (a, b) in a_data.iter().zip(b_data) {
                *b = f(*a, *b);
            }
        }
    } else if ash.contains(&0) || bsh.contains(&0) {
        if ash.len() > bsh.len() {
            *b = a;
        }
    } else {
        match ash.len().cmp(&bsh.len()) {
            Ordering::Greater => {
                let a_data = a.data.as_mut_slice();
                let b_data = b.data.as_slice();
                bin_pervade_recursive_mut_left(a_data, ash, b_data, bsh, f);
                *b = a;
            }
            Ordering::Less => {
                let a_data = a.data.as_slice();
                let b_data = b.data.as_mut_slice();
                bin_pervade_recursive_mut_right(a_data, ash, b_data, bsh, f);
            }
            Ordering::Equal => {
                let a_data = a.data.as_mut_slice();
                let b_data = b.data.as_mut_slice();
                bin_pervade_recursive_mut(a_data, ash, b_data, bsh, f);
            }
        }
    }
    Ok(())
}

fn bin_pervade_recursive_mut<T>(
    a_data: &mut [T],
    a_shape: &[usize],
    b_data: &mut [T],
    b_shape: &[usize],
    f: impl Fn(T, T) -> T + Copy,
) where
    T: Copy,
{
    match (a_shape, b_shape) {
        ([], []) => {
            panic!("should never call `bin_pervade_recursive_mut` with scalars")
        }
        (_, []) => {
            let b_scalar = b_data[0];
            for a in a_data {
                *a = f(*a, b_scalar);
            }
        }
        ([], _) => {
            let a_scalar = a_data[0];
            for b in b_data {
                *b = f(a_scalar, *b);
            }
        }
        (ash, bsh) => {
            let a_row_len = a_data.len() / ash[0];
            let b_row_len = b_data.len() / bsh[0];
            for (a, b) in a_data
                .chunks_exact_mut(a_row_len)
                .zip(b_data.chunks_exact_mut(b_row_len))
            {
                bin_pervade_recursive_mut(a, &ash[1..], b, &bsh[1..], f);
            }
        }
    }
}

fn bin_pervade_recursive_mut_left<T>(
    a_data: &mut [T],
    a_shape: &[usize],
    b_data: &[T],
    b_shape: &[usize],
    f: impl Fn(T, T) -> T + Copy,
) where
    T: Copy,
{
    match (a_shape, b_shape) {
        ([], _) => a_data[0] = f(a_data[0], b_data[0]),
        (_, []) => {
            let b_scalar = b_data[0];
            for a in a_data {
                *a = f(*a, b_scalar);
            }
        }
        (ash, bsh) => {
            let a_row_len = a_data.len() / ash[0];
            let b_row_len = b_data.len() / bsh[0];
            for (a, b) in a_data
                .chunks_exact_mut(a_row_len)
                .zip(b_data.chunks_exact(b_row_len))
            {
                bin_pervade_recursive_mut_left(a, &ash[1..], b, &bsh[1..], f);
            }
        }
    }
}

fn bin_pervade_recursive_mut_right<T>(
    a_data: &[T],
    a_shape: &[usize],
    b_data: &mut [T],
    b_shape: &[usize],
    f: impl Fn(T, T) -> T + Copy,
) where
    T: Copy,
{
    match (a_shape, b_shape) {
        (_, []) => b_data[0] = f(a_data[0], b_data[0]),
        ([], _) => {
            let a_scalar = a_data[0];
            for b in b_data {
                *b = f(a_scalar, *b);
            }
        }
        (ash, bsh) => {
            let a_row_len = a_data.len() / ash[0];
            let b_row_len = b_data.len() / bsh[0];
            for (a, b) in a_data
                .chunks_exact(a_row_len)
                .zip(b_data.chunks_exact_mut(b_row_len))
            {
                bin_pervade_recursive_mut_right(a, &ash[1..], b, &bsh[1..], f);
            }
        }
    }
}

pub mod not {
    use super::*;
    pub fn num(a: f64) -> f64 {
        1.0 - a
    }
    pub fn byte(a: u8) -> f64 {
        num(a.into())
    }
    pub fn bool(a: u8) -> u8 {
        a ^ 1u8
    }
    pub fn com(a: Complex) -> Complex {
        1.0 - a
    }
    pub fn error<T: Display>(a: T, env: &Uiua) -> UiuaError {
        env.error(format!("Cannot not {a}"))
    }
}

fn toggle_char_case(a: char) -> char {
    if a.is_lowercase() {
        let mut upper = a.to_uppercase();
        if upper.len() == 1 {
            upper.next().unwrap()
        } else {
            a
        }
    } else if a.is_uppercase() {
        let mut lower = a.to_lowercase();
        if lower.len() == 1 {
            lower.next().unwrap()
        } else {
            a
        }
    } else {
        a
    }
}

pub mod scalar_neg {
    use super::*;
    pub fn num(a: f64) -> f64 {
        -a
    }
    pub fn byte(a: u8) -> f64 {
        -f64::from(a)
    }
    pub fn char(a: char) -> char {
        toggle_char_case(a)
    }
    pub fn com(a: Complex) -> Complex {
        -a
    }
    pub fn error<T: Display>(a: T, env: &Uiua) -> UiuaError {
        env.error(format!("Cannot negate {a}"))
    }
}
pub mod scalar_abs {
    use super::*;
    pub fn num(a: f64) -> f64 {
        a.abs()
    }
    pub fn byte(a: u8) -> u8 {
        a
    }
    pub fn char(a: char) -> char {
        if a.is_lowercase() {
            let mut upper = a.to_uppercase();
            if upper.len() == 1 {
                upper.next().unwrap()
            } else {
                a
            }
        } else {
            a
        }
    }
    pub fn com(a: Complex) -> f64 {
        a.abs()
    }
    pub fn error<T: Display>(a: T, env: &Uiua) -> UiuaError {
        env.error(format!("Cannot take the absolute value of {a}"))
    }
}

fn character_sign(a: char) -> f64 {
    if a.is_uppercase() {
        1.0
    } else if a.is_lowercase() {
        -1.0
    } else {
        0.0
    }
}

pub mod sign {
    use super::*;
    pub fn num(a: f64) -> f64 {
        if a.is_nan() {
            f64::NAN
        } else if a == 0.0 {
            0.0
        } else {
            a.signum()
        }
    }
    pub fn byte(a: u8) -> u8 {
        (a > 0) as u8
    }
    pub fn char(a: char) -> f64 {
        character_sign(a)
    }
    pub fn com(a: Complex) -> Complex {
        a.normalize()
    }
    pub fn error<T: Display>(a: T, env: &Uiua) -> UiuaError {
        env.error(format!("Cannot get the sign of {a}"))
    }
}
pub mod sqrt {
    use super::*;
    pub fn num(a: f64) -> f64 {
        a.sqrt()
    }
    pub fn byte(a: u8) -> f64 {
        f64::from(a).sqrt()
    }
    pub fn bool(a: u8) -> u8 {
        a
    }
    pub fn com(a: Complex) -> Complex {
        a.sqrt()
    }
    pub fn error<T: Display>(a: T, env: &Uiua) -> UiuaError {
        env.error(format!("Cannot take the square root of {a}"))
    }
}
pub mod sin {
    use super::*;
    pub fn num(a: f64) -> f64 {
        a.sin()
    }
    pub fn byte(a: u8) -> f64 {
        f64::from(a).sin()
    }
    pub fn com(a: Complex) -> Complex {
        a.sin()
    }
    pub fn error<T: Display>(a: T, env: &Uiua) -> UiuaError {
        env.error(format!("Cannot get the sine of {a}"))
    }
}
pub mod cos {
    use super::*;
    pub fn num(a: f64) -> f64 {
        a.cos()
    }
    pub fn byte(a: u8) -> f64 {
        f64::from(a).cos()
    }
    pub fn com(a: Complex) -> Complex {
        a.cos()
    }
    pub fn error<T: Display>(a: T, env: &Uiua) -> UiuaError {
        env.error(format!("Cannot get the cosine of {a}"))
    }
}
pub mod asin {
    use super::*;
    pub fn num(a: f64) -> f64 {
        a.asin()
    }
    pub fn byte(a: u8) -> f64 {
        f64::from(a).asin()
    }
    pub fn com(a: Complex) -> Complex {
        a.asin()
    }
    pub fn error<T: Display>(a: T, env: &Uiua) -> UiuaError {
        env.error(format!("Cannot get the arcsine of {a}"))
    }
}
pub mod floor {
    use super::*;
    pub fn num(a: f64) -> f64 {
        a.floor()
    }
    pub fn byte(a: u8) -> u8 {
        a
    }
    pub fn com(a: Complex) -> Complex {
        a.floor()
    }
    pub fn error<T: Display>(a: T, env: &Uiua) -> UiuaError {
        env.error(format!("Cannot get the floor of {a}"))
    }
}
pub mod ceil {
    use super::*;
    pub fn num(a: f64) -> f64 {
        a.ceil()
    }
    pub fn byte(a: u8) -> u8 {
        a
    }
    pub fn com(a: Complex) -> Complex {
        a.ceil()
    }
    pub fn error<T: Display>(a: T, env: &Uiua) -> UiuaError {
        env.error(format!("Cannot get the ceiling of {a}"))
    }
}
pub mod round {
    use super::*;
    pub fn num(a: f64) -> f64 {
        a.round()
    }
    pub fn byte(a: u8) -> u8 {
        a
    }
    pub fn com(a: Complex) -> Complex {
        a.round()
    }
    pub fn error<T: Display>(a: T, env: &Uiua) -> UiuaError {
        env.error(format!("Cannot get the rounded value of {a}"))
    }
}

pub mod complex_re {
    use super::*;

    pub fn com(a: Complex) -> f64 {
        a.re
    }
    pub fn generic<T>(a: T) -> T {
        a
    }
    pub fn error<T: Display>(a: T, env: &Uiua) -> UiuaError {
        env.error(format!("Cannot get the real part of {a}"))
    }
}
pub mod complex_im {
    use super::*;

    pub fn com(a: Complex) -> f64 {
        a.im
    }
    pub fn num(_a: f64) -> f64 {
        0.0
    }
    pub fn byte(_a: u8) -> u8 {
        0
    }
    pub fn error<T: Display>(a: T, env: &Uiua) -> UiuaError {
        env.error(format!("Cannot get the imaginary part of {a}"))
    }
}

macro_rules! eq_impl {
    ($name:ident $eq:tt $ordering:expr) => {
        pub mod $name {
            use super::*;
            pub fn always_greater<A, B>(_: A, _: B) -> u8 {
                ($ordering $eq Ordering::Less).into()
            }
            pub fn always_less<A, B>(_: A, _: B) -> u8 {
                ($ordering $eq Ordering::Greater).into()
            }
            pub fn num_num(a: f64, b: f64) -> u8 {
                (b.array_cmp(&a) $eq $ordering) as u8
            }
            pub fn com_x(a: Complex, b: impl Into<Complex>) -> u8 {
                (b.into().array_cmp(&a) $eq $ordering) as u8
            }
            pub fn x_com(a: impl Into<Complex>, b: Complex) -> u8 {
                (b.array_cmp(&a.into()) $eq $ordering) as u8
            }
                        pub fn byte_num(a: u8, b: f64) -> u8 {
                (b.array_cmp(&f64::from(a)) $eq $ordering) as u8
            }
                        pub fn num_byte(a: f64, b: u8) -> u8 {
                (f64::from(b).array_cmp(&a) $eq $ordering) as u8
            }
            pub fn generic<T: Ord>(a: T, b: T) -> u8 {
                (b.cmp(&a) $eq $ordering).into()
            }
            pub fn same_type<T: ArrayCmp + From<u8>>(a: T, b: T) -> T {
               ((b.array_cmp(&a) $eq $ordering) as u8).into()
            }
            pub fn error<T: Display>(a: T, b: T, _env: &Uiua) -> UiuaError {
                unreachable!("Comparisons cannot fail, failed to compare {a} and {b}")
            }
        }
    };
}

macro_rules! cmp_impl {
    ($name:ident $eq:tt $ordering:expr) => {
        pub mod $name {
            use super::*;
            pub fn always_greater<A, B>(_: A, _: B) -> u8 {
                ($ordering $eq Ordering::Less).into()
            }
            pub fn always_less<A, B>(_: A, _: B) -> u8 {
                ($ordering $eq Ordering::Greater).into()
            }
            pub fn num_num(a: f64, b: f64) -> u8 {
                (b.array_cmp(&a) $eq $ordering) as u8
            }
            pub fn com_x(a: Complex, b: impl Into<Complex>) -> Complex {
                let b = b.into();
                Complex::new(
                    (b.re.array_cmp(&a.re) $eq $ordering) as u8 as f64,
                    (b.im.array_cmp(&a.im) $eq $ordering) as u8 as f64
                )
            }
            pub fn x_com(a: impl Into<Complex>, b: Complex) -> Complex {
                let a = a.into();
                Complex::new(
                    (b.re.array_cmp(&a.re) $eq $ordering) as u8 as f64,
                    (b.im.array_cmp(&a.im) $eq $ordering) as u8 as f64
                )
            }
                        pub fn byte_num(a: u8, b: f64) -> u8 {
                (b.array_cmp(&f64::from(a)) $eq $ordering) as u8
            }
                        pub fn num_byte(a: f64, b: u8) -> u8 {
                (f64::from(b).array_cmp(&a) $eq $ordering) as u8
            }
            pub fn generic<T: Ord>(a: T, b: T) -> u8 {
                (b.cmp(&a) $eq $ordering).into()
            }
            pub fn same_type<T: ArrayCmp + From<u8>>(a: T, b: T) -> T {
               ((b.array_cmp(&a) $eq $ordering) as u8).into()
            }
            pub fn error<T: Display>(a: T, b: T, _env: &Uiua) -> UiuaError {
                unreachable!("Comparisons cannot fail, failed to compare {a} and {b}")
            }
        }
    };
}

eq_impl!(is_eq == Ordering::Equal);
eq_impl!(is_ne != Ordering::Equal);
cmp_impl!(is_lt == Ordering::Less);
cmp_impl!(is_le != Ordering::Greater);
cmp_impl!(is_gt == Ordering::Greater);
cmp_impl!(is_ge != Ordering::Less);

pub mod add {
    use super::*;
    pub fn num_num(a: f64, b: f64) -> f64 {
        b + a
    }
    pub fn byte_byte(a: u8, b: u8) -> f64 {
        f64::from(a) + f64::from(b)
    }
    pub fn bool_bool(a: u8, b: u8) -> u8 {
        b + a
    }
    pub fn byte_num(a: u8, b: f64) -> f64 {
        b + f64::from(a)
    }
    pub fn num_byte(a: f64, b: u8) -> f64 {
        a + f64::from(b)
    }
    pub fn com_x(a: Complex, b: impl Into<Complex>) -> Complex {
        b.into() + a
    }
    pub fn x_com(a: impl Into<Complex>, b: Complex) -> Complex {
        b + a.into()
    }
    pub fn num_char(a: f64, b: char) -> char {
        char::from_u32((b as i64 + a as i64) as u32).unwrap_or('\0')
    }
    pub fn char_num(a: char, b: f64) -> char {
        char::from_u32((b as i64 + a as i64) as u32).unwrap_or('\0')
    }
    pub fn byte_char(a: u8, b: char) -> char {
        char::from_u32((b as i64 + a as i64) as u32).unwrap_or('\0')
    }
    pub fn char_byte(a: char, b: u8) -> char {
        char::from_u32((b as i64 + a as i64) as u32).unwrap_or('\0')
    }
    pub fn error<T: Display>(a: T, b: T, env: &Uiua) -> UiuaError {
        env.error(format!("Cannot add {a} and {b}"))
    }
}

pub mod sub {
    use super::*;
    pub fn num_num(a: f64, b: f64) -> f64 {
        b - a
    }
    pub fn byte_byte(a: u8, b: u8) -> f64 {
        f64::from(b) - f64::from(a)
    }
    pub fn byte_num(a: u8, b: f64) -> f64 {
        b - f64::from(a)
    }
    pub fn num_byte(a: f64, b: u8) -> f64 {
        f64::from(b) - a
    }
    pub fn com_x(a: Complex, b: impl Into<Complex>) -> Complex {
        b.into() - a
    }
    pub fn x_com(a: impl Into<Complex>, b: Complex) -> Complex {
        b - a.into()
    }
    pub fn num_char(a: f64, b: char) -> char {
        char::from_u32(((b as i64) - (a as i64)) as u32).unwrap_or('\0')
    }
    pub fn char_char(a: char, b: char) -> f64 {
        ((b as i64) - (a as i64)) as f64
    }
    pub fn byte_char(a: u8, b: char) -> char {
        char::from_u32(((b as i64) - (a as i64)) as u32).unwrap_or('\0')
    }
    pub fn error<T: Display>(a: T, b: T, env: &Uiua) -> UiuaError {
        env.error(format!("Cannot subtract {a} from {b}"))
    }
}

macro_rules! bin_op_mod {
    ($name:ident, $a:ident, $b:ident, $byte_convert:expr, $byte_ret:ty, $f:expr, $err:literal) => {
        pub mod $name {
            use super::*;
            pub fn num_num($a: f64, $b: f64) -> f64 {
                $f
            }
            pub fn byte_byte($a: u8, $b: u8) -> f64 {
                let $a = $byte_convert($a);
                let $b = $byte_convert($b);
                $f
            }
            pub fn byte_num($a: u8, $b: f64) -> f64 {
                let $a = $byte_convert($a);
                $f
            }
            pub fn num_byte($a: f64, $b: u8) -> f64 {
                let $b = $byte_convert($b);
                $f
            }

            pub fn com_x($a: Complex, $b: impl Into<Complex>) -> Complex {
                let $b = $b.into();
                $f
            }

            pub fn x_com($a: impl Into<Complex>, $b: Complex) -> Complex {
                let $a = $a.into();
                $f
            }
            pub fn error<T: Display>($a: T, $b: T, env: &Uiua) -> UiuaError {
                env.error(format!($err))
            }
        }
    };
}

pub mod mul {
    use super::*;
    pub fn num_num(a: f64, b: f64) -> f64 {
        b * a
    }
    pub fn byte_byte(a: u8, b: u8) -> f64 {
        f64::from(b) * f64::from(a)
    }
    pub fn bool_bool(a: u8, b: u8) -> u8 {
        b & a
    }
    pub fn byte_num(a: u8, b: f64) -> f64 {
        b * f64::from(a)
    }
    pub fn num_byte(a: f64, b: u8) -> f64 {
        f64::from(b) * a
    }
    pub fn num_char(a: f64, b: char) -> char {
        if a < 0.0 {
            toggle_char_case(b)
        } else {
            b
        }
    }
    pub fn char_num(a: char, b: f64) -> char {
        if b < 0.0 {
            toggle_char_case(a)
        } else {
            a
        }
    }
    pub fn byte_char(_: u8, b: char) -> char {
        b
    }
    pub fn char_byte(a: char, _: u8) -> char {
        a
    }
    pub fn com_x(a: Complex, b: impl Into<Complex>) -> Complex {
        b.into() * a
    }
    pub fn x_com(a: impl Into<Complex>, b: Complex) -> Complex {
        b * a.into()
    }
    pub fn error<T: Display>(a: T, b: T, env: &Uiua) -> UiuaError {
        env.error(format!("Cannot multiply {a} and {b}"))
    }
}

pub mod div {
    use super::*;
    pub fn num_num(a: f64, b: f64) -> f64 {
        b / a
    }
    pub fn byte_byte(a: u8, b: u8) -> f64 {
        f64::from(b) / f64::from(a)
    }
    pub fn byte_num(a: u8, b: f64) -> f64 {
        b / f64::from(a)
    }
    pub fn num_byte(a: f64, b: u8) -> f64 {
        f64::from(b) / a
    }
    pub fn num_char(a: f64, b: char) -> char {
        if a < 0.0 {
            toggle_char_case(b)
        } else {
            b
        }
    }
    pub fn byte_char(_: u8, b: char) -> char {
        b
    }
    pub fn com_x(a: Complex, b: impl Into<Complex>) -> Complex {
        b.into() / a
    }
    pub fn x_com(a: impl Into<Complex>, b: Complex) -> Complex {
        b / a.into()
    }
    pub fn error<T: Display>(a: T, b: T, env: &Uiua) -> UiuaError {
        env.error(format!("Cannot divide {b} by {a}"))
    }
}

pub mod modulus {
    use super::*;
    pub fn num_num(a: f64, b: f64) -> f64 {
        b.rem_euclid(a)
    }
    pub fn byte_byte(a: u8, b: u8) -> f64 {
        num_num(a.into(), b.into())
    }
    pub fn byte_num(a: u8, b: f64) -> f64 {
        num_num(a.into(), b)
    }
    pub fn num_byte(a: f64, b: u8) -> f64 {
        num_num(a, b.into())
    }
    pub fn com_com(a: Complex, b: Complex) -> Complex {
        b % a
    }
    pub fn com_x(a: Complex, b: impl Into<Complex>) -> Complex {
        b.into() % a
    }
    pub fn x_com(a: impl Into<f64>, b: Complex) -> Complex {
        b % a.into()
    }
    pub fn error<T: Display>(a: T, b: T, env: &Uiua) -> UiuaError {
        env.error(format!("Cannot modulo {a} and {b}"))
    }
}
bin_op_mod!(
    atan2,
    a,
    b,
    f64::from,
    f64,
    a.atan2(b),
    "Cannot get the atan2 of {a} and {b}"
);
pub mod pow {
    use super::*;
    pub fn num_num(a: f64, b: f64) -> f64 {
        b.powf(a)
    }
    pub fn byte_byte(a: u8, b: u8) -> f64 {
        f64::from(b).powf(f64::from(a))
    }
    pub fn byte_num(a: u8, b: f64) -> f64 {
        b.powi(a as i32)
    }
    pub fn num_byte(a: f64, b: u8) -> f64 {
        f64::from(b).powf(a)
    }
    pub fn com_x(a: Complex, b: impl Into<Complex>) -> Complex {
        b.into().powc(a)
    }
    pub fn x_com(a: impl Into<Complex>, b: Complex) -> Complex {
        b.powc(a.into())
    }
    pub fn error<T: Display>(a: T, b: T, env: &Uiua) -> UiuaError {
        env.error(format!("Cannot get the power of {a} to {b}"))
    }
}
bin_op_mod!(
    log,
    a,
    b,
    f64::from,
    f64,
    b.log(a),
    "Cannot get the log base {b} of {a}"
);
pub mod complex {
    use super::*;

    pub fn num_num(a: f64, b: f64) -> Complex {
        Complex::new(b, a)
    }
    pub fn byte_byte(a: u8, b: u8) -> Complex {
        Complex::new(b.into(), a.into())
    }
    pub fn byte_num(a: u8, b: f64) -> Complex {
        Complex::new(b, a.into())
    }
    pub fn num_byte(a: f64, b: u8) -> Complex {
        Complex::new(b.into(), a)
    }
    pub fn com_x(a: Complex, b: impl Into<Complex>) -> Complex {
        b.into() + a * Complex::I
    }
    pub fn x_com(a: impl Into<Complex>, b: Complex) -> Complex {
        b + a.into() * Complex::I
    }
    pub fn error<T: Display>(a: T, b: T, env: &Uiua) -> UiuaError {
        env.error(format!("Cannot get the form a complex number with {b} as the real part and {a} as the imaginary part"))
    }
}

pub mod max {
    use super::*;
    pub fn num_num(a: f64, b: f64) -> f64 {
        a.max(b)
    }
    pub fn byte_byte(a: u8, b: u8) -> u8 {
        a.max(b)
    }
    pub fn bool_bool(a: u8, b: u8) -> u8 {
        a | b
    }
    pub fn char_char(a: char, b: char) -> char {
        a.max(b)
    }
    pub fn num_byte(a: f64, b: u8) -> f64 {
        num_num(a, b.into())
    }
    pub fn byte_num(a: u8, b: f64) -> f64 {
        num_num(a.into(), b)
    }
    pub fn com_x(a: Complex, b: impl Into<Complex>) -> Complex {
        a.max(b.into())
    }
    pub fn x_com(a: impl Into<Complex>, b: Complex) -> Complex {
        a.into().max(b)
    }
    pub fn error<T: Display>(a: T, b: T, env: &Uiua) -> UiuaError {
        env.error(format!("Cannot get the max of {a} and {b}"))
    }
}

pub mod min {
    use super::*;
    pub fn num_num(a: f64, b: f64) -> f64 {
        a.min(b)
    }
    pub fn byte_byte(a: u8, b: u8) -> u8 {
        a.min(b)
    }
    pub fn bool_bool(a: u8, b: u8) -> u8 {
        a & b
    }
    pub fn char_char(a: char, b: char) -> char {
        a.min(b)
    }
    pub fn num_byte(a: f64, b: u8) -> f64 {
        num_num(a, b.into())
    }
    pub fn byte_num(a: u8, b: f64) -> f64 {
        num_num(a.into(), b)
    }
    pub fn com_x(a: Complex, b: impl Into<Complex>) -> Complex {
        a.min(b.into())
    }
    pub fn x_com(a: impl Into<Complex>, b: Complex) -> Complex {
        a.into().min(b)
    }
    pub fn error<T: Display>(a: T, b: T, env: &Uiua) -> UiuaError {
        env.error(format!("Cannot get the min of {a} and {b}"))
    }
}
