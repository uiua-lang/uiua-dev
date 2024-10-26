use super::*;

impl Node {
    /// Get both parts of this node's under inverse
    pub fn under_inverse(
        &self,
        _g_sig: Signature,
        _asm: &Assembly,
    ) -> InversionResult<(Node, Node)> {
        dbgln!("under-inverting {self:?}");
        todo!("new under inversion")
    }
}
