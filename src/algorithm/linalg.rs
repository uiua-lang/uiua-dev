//! Linear algebra implementations

use ecow::eco_vec;

use crate::{Array, RealArrayValue, Shape, Uiua, UiuaResult, Value};

use super::{
    pervade::add,
    reduce::{fast_reduce, fast_reduce_different},
    FillContext,
};

impl Value {
    /// Multiply this value as a matrix by another
    pub fn matrix_product(self, other: &Self, env: &mut Uiua) -> UiuaResult<Value> {
        if !(self.rank() == 2 && other.rank() == 2) {
            let (mut xs, ys) = (self, other);
            xs.transpose();
            let mut new_shape = Shape::from([xs.row_count(), ys.row_count()]);
            let mut builder = Value::builder(xs.row_count() * ys.row_count());
            env.without_fill(|env| -> UiuaResult {
                for x_row in xs.rows() {
                    for y_row in ys.rows() {
                        let prod = x_row.clone().mul(y_row, 0, 0, env)?;
                        let sum: Value = match prod {
                            Value::Num(num) => {
                                fast_reduce(num, 0.0, env.scalar_fill().ok(), 0, add::num_num)
                                    .into()
                            }
                            Value::Byte(byte) => fast_reduce_different(
                                byte,
                                0.0,
                                env.scalar_fill().ok(),
                                0,
                                add::num_num,
                                add::num_byte,
                            )
                            .into(),
                            Value::Complex(comp) => {
                                fast_reduce(comp, 0.0.into(), env.scalar_fill().ok(), 0, add::com_x)
                                    .into()
                            }
                            Value::Char(_) => {
                                return Err(env.error("Cannot add characters to characters"))
                            }
                            val => {
                                let mut sum: Value = 0u8.into();
                                for val in val.into_rows() {
                                    sum = sum.add(val, 0, 0, env)?;
                                }
                                sum
                            }
                        };
                        builder.add_row(sum, env)?;
                    }
                }
                Ok(())
            })?;
            let mut tabled = builder.finish();
            new_shape.extend_from_slice(&tabled.shape()[1..]);
            *tabled.shape_mut() = new_shape;
            tabled.transpose();
            return Ok(tabled);
        }
        match (self, other) {
            (Value::Num(a), Value::Num(b)) => a.matrix_product(b, env),
            (Value::Byte(a), Value::Byte(b)) => a.matrix_product(b, env),
            (Value::Num(a), Value::Byte(b)) => a.matrix_product(b, env),
            (Value::Byte(a), Value::Num(b)) => a.matrix_product(b, env),
            (a, b) => Err(env.error(format!(
                "Cannot matrix-multiply {} and {}",
                a.type_name_plural(),
                b.type_name_plural()
            ))),
        }
        .map(Into::into)
    }
}

impl<T: RealArrayValue> Array<T> {
    /// Multiply this array as a matrix by another
    fn matrix_product<U: RealArrayValue>(
        &self,
        rhs: &Array<U>,
        env: &Uiua,
    ) -> UiuaResult<Array<f64>> {
        let (a, b) = (rhs, self);
        assert_eq!((2, 2), (a.rank(), b.rank()));
        let a_row_count = a.row_count();
        let a_col_count = a.shape.row().row_count();
        let b_row_count = b.row_count();
        let b_col_count = b.shape.row().row_count();
        if a_col_count != b_row_count {
            return Err(env.error(format!(
                "Cannot matrix-multiply arrays with shapes {} and {}",
                a.shape, b.shape
            )));
        }
        let a_row_len = a.row_len();
        let b_row_len = b.row_len();
        let a_data = a.data.as_slice();
        let b_data = b.data.as_slice();
        let mut new_data = eco_vec![0.0; a_row_count * b_col_count ];
        let new_slice = new_data.make_mut();
        for i in 0..a_row_count {
            for j in 0..b_col_count {
                for k in 0..a_col_count {
                    let a = a_data[i * a_row_len + k].into_f64();
                    let b = b_data[k * b_row_len + j].into_f64();
                    new_slice[i * b_col_count + j] += a * b;
                }
            }
        }
        Ok(Array::new([a_row_count, b_col_count], new_data))
    }
}
