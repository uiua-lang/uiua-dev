//! Linear algebra implementations

use ecow::eco_vec;

use crate::{
    algorithm::pervade::{self, bin_pervade_recursive, InfalliblePervasiveFn},
    Array, Shape, Uiua, UiuaResult,
};

use super::shape_prefixes_match;

impl Array<f64> {
    /// Multiply this array as a matrix by another
    pub fn matrix_product(&self, rhs: &Self, env: &Uiua) -> UiuaResult<Self> {
        let (a, b) = (self, rhs);
        let a_cell_rank = a.rank().saturating_sub(2);
        let b_cell_rank = b.rank().saturating_sub(2);
        let a_cell_shape = Shape::from(&a.shape[a_cell_rank..]);
        let b_cell_shape = Shape::from(&a.shape[b_cell_rank..]);
        if !shape_prefixes_match(&a_cell_shape, &b_cell_shape) {
            return Err(env.error(format!(
                "Cannot matrix-multiply arrays with shapes {} and {}",
                a.shape, b.shape
            )));
        }
        let a_row_count = a.row_count();
        let a_col_count = a.shape.row().row_count();
        let _b_row_count = b.row_count();
        let b_col_count = b.shape.row().row_count();
        let a_row_len = a.row_len();
        let b_row_len = b.row_len();
        let a_cell_len = a_cell_shape.elements();
        let b_cell_len = b_cell_shape.elements();
        let a_data = a.data.as_slice();
        let b_data = b.data.as_slice();
        let mut new_data = eco_vec![0.0; a_row_len * b_row_len * a_cell_len.max(b_cell_len)];
        let new_slice = new_data.make_mut();
        let mut cell_prod = vec![0.0; a_cell_len.max(b_cell_len)];
        for i in 0..a_row_count {
            for j in 0..b_col_count {
                for k in 0..a_col_count {
                    let a_start = i * a_row_len + k * a_cell_len;
                    let b_start = k * b_row_len + j * b_cell_len;
                    let a_cell = &a_data[a_start..a_start + a_cell_len];
                    let b_cell = &b_data[b_start..b_start + b_cell_len];
                    bin_pervade_recursive(
                        &(&*a_cell_shape, a_cell),
                        &(&*b_cell_shape, b_cell),
                        &mut cell_prod,
                        env,
                        InfalliblePervasiveFn::new(pervade::mul::num_num),
                    )?;
                    let c_start = i * b_row_len + j * a_cell_len;
                    for (prod, c) in new_slice[c_start..].iter_mut().zip(&cell_prod) {
                        *prod += c;
                    }
                }
            }
        }
        let cell_shape = if a_cell_len > b_cell_len {
            a_cell_shape
        } else {
            b_cell_shape
        };
        let mut new_shape = cell_shape;
        new_shape.insert(0, b_col_count);
        new_shape.insert(0, a_row_count);
        Ok(Array::new(new_shape, new_data))
    }
}
