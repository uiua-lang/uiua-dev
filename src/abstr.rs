use std::{
    borrow::Cow,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use ecow::eco_vec;
use serde::*;

use crate::{Shape, Uiua, UiuaResult, Value};

/// A trait for abstract arrays
pub trait AbstractArray: Send + Sync + 'static {
    /// The name of the abstraction
    fn name(&self) -> Cow<'static, str>;
    /// The type number
    fn type_num(&self) -> Option<u8>;
    /// The length of the array
    fn length(&self) -> Option<usize>;
    /// The shape of the array's rows
    fn row_shape(&self) -> Shape;
    /// Clone the array
    fn clone(&self) -> Arc<dyn AbstractArray>;
    /// A static tag for serialization
    fn tag(&self) -> &'static str;
    /// Fields for serialization
    fn fields(&self) -> Vec<String> {
        Vec::new()
    }
    /// Materialize the array into a concrete value
    fn materialize(&mut self, env: &Uiua) -> UiuaResult<Value> {
        Err(env.error(format!("{} cannot be materialized", self.name())))
    }
    /// The shape of the array
    fn shape(&self) -> Option<Shape> {
        let len = self.length()?;
        let mut shape = self.row_shape();
        shape.insert(0, len);
        Some(shape)
    }
    /// The rank of the array
    fn rank(&self) -> usize {
        self.row_shape().len() + 1
    }
    /// A value representing the shape of the array
    fn shape_value(&self) -> Value {
        let mut data = eco_vec![self.length().map(|l| l as f64).unwrap_or(f64::INFINITY)];
        data.extend(self.row_shape().iter().map(|&d| d as f64));
        Value::from(data)
    }
    /// A value representing the length of the array
    fn length_value(&self) -> Value {
        self.length()
            .map(|l| l as f64)
            .unwrap_or(f64::INFINITY)
            .into()
    }
}

/// An abstract array
#[derive(Clone)]
pub struct Abstract(Arc<dyn AbstractArray>);

impl<'de> Deserialize<'de> for Abstract {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let rep = AbstractRep::deserialize(deserializer)?;
        Ok(match rep.tag.as_str() {
            "INFINITE_RANGE" => InfiniteRange.into(),
            tag => {
                return Err(de::Error::custom(format!(
                    "Unknown abstract array tag {tag}"
                )))
            }
        })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct InfiniteRange;

impl AbstractArray for InfiniteRange {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("⇡∞")
    }
    fn type_num(&self) -> Option<u8> {
        Some(0)
    }
    fn length(&self) -> Option<usize> {
        None
    }
    fn row_shape(&self) -> Shape {
        Shape::scalar()
    }
    fn clone(&self) -> Arc<dyn AbstractArray> {
        Arc::new(*self)
    }
    fn tag(&self) -> &'static str {
        "INFINITE_RANGE"
    }
}

impl<A: AbstractArray> From<A> for Abstract {
    fn from(a: A) -> Self {
        Self(Arc::new(a))
    }
}

impl Deref for Abstract {
    type Target = dyn AbstractArray;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl DerefMut for Abstract {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if Arc::strong_count(&self.0) > 1 {
            self.0 = self.0.clone();
        }
        Arc::get_mut(&mut self.0).unwrap()
    }
}

#[derive(Serialize, Deserialize)]
struct AbstractRep {
    tag: String,
    fields: Vec<String>,
}

impl Serialize for Abstract {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let tag = self.0.tag();
        let fields = self.0.fields();
        AbstractRep {
            tag: tag.into(),
            fields,
        }
        .serialize(serializer)
    }
}
