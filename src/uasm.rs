use std::{path::Path, sync::Arc};

use ecow::EcoString;

use crate::{CodeSpan, InputSrc, LexError, Loc, Sp};

#[derive(Clone)]
enum UasmToken {
    Ident,
    Number,
    String(EcoString),
    Newline,
    Colon,
}

struct Lexer {
    chars: Vec<char>,
    loc: Loc,
    src: Arc<Path>,
    tokens: Vec<Sp<UasmToken>>,
    errors: Vec<Sp<LexError>>,
}

impl Lexer {
    fn peek_char(&self) -> Option<char> {
        self.chars.get(self.loc.char_pos as usize).copied()
    }
    fn update_loc(&mut self, c: char) {
        for c in &self.chars {
            match c {
                '\n' => {
                    self.loc.line += 1;
                    self.loc.col = 1;
                }
                '\r' => {}
                _ => self.loc.col += 1,
            }
        }
        self.loc.char_pos += 1;
        self.loc.byte_pos += c.len_utf8() as u32;
    }
    fn next_char_if(&mut self, f: impl Fn(char) -> bool) -> Option<char> {
        let c = *self.chars.get(self.loc.char_pos as usize)?;
        if !f(c) {
            return None;
        }
        self.update_loc(c);
        Some(c)
    }
    fn next_char(&mut self) -> Option<char> {
        self.next_char_if(|_| true)
    }
    fn next_char_exact(&mut self, c: char) -> bool {
        self.next_char_if(|c2| c2 == c).is_some()
    }
    fn make_span(&self, start: Loc, end: Loc) -> CodeSpan {
        CodeSpan {
            start,
            end,
            src: InputSrc::File(self.src.clone()),
        }
    }
    fn end_span(&self, start: Loc) -> CodeSpan {
        assert!(self.loc.char_pos >= start.char_pos, "empty span");
        self.make_span(start, self.loc)
    }
    fn end(&mut self, token: impl Into<UasmToken>, start: Loc) {
        self.tokens.push(Sp {
            value: token.into(),
            span: self.end_span(start),
        })
    }
    fn run(mut self) -> Result<Vec<Sp<UasmToken>>, Sp<LexError>> {
        loop {
            let start = self.loc;
            let Some(c) = self.next_char() else {
                break;
            };
            match c {
                ':' => self.end(UasmToken::Colon, start),
                '\n' => self.end(UasmToken::Newline, start),
                c if c.is_alphabetic() => {
                    while self.next_char_if(char::is_alphabetic).is_some() {}
                    self.end(UasmToken::Ident, start);
                }
                c if c == '-' || c.is_ascii_digit() => {
                    while self.next_char_if(|c| c.is_ascii_digit()).is_some() {}
                    if self.next_char_exact('.') {
                        while self.next_char_if(|c| c.is_ascii_digit()).is_some() {}
                    }
                    self.end(UasmToken::Number, start);
                }
                c if c.is_whitespace() => {}
                c => {
                    return Err(self
                        .end_span(start)
                        .sp(LexError::UnexpectedChar(c.to_string())))
                }
            }
        }
        Ok(self.tokens)
    }
}
