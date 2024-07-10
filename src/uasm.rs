use std::{path::Path, sync::Arc};

use ecow::EcoString;

use crate::{Assembly, CodeSpan, InputSrc, Instr, LexError, Loc, ParseError, Sp};

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
                // String
                '"' => {
                    let mut s = EcoString::new();
                    let mut escaped = false;
                    while let Some(c) = self.next_char() {
                        if escaped {
                            match c {
                                'n' => s.push('\n'),
                                'r' => s.push('\r'),
                                't' => s.push('\t'),
                                '"' => s.push('"'),
                                '\\' => s.push('\\'),
                                _ => {
                                    return Err(self
                                        .end_span(start)
                                        .sp(LexError::InvalidEscape(c.into())))
                                }
                            }
                            escaped = false;
                        } else if c == '\\' {
                            escaped = true;
                        } else if c == '"' {
                            break;
                        } else {
                            s.push(c);
                        }
                    }
                    self.end(UasmToken::String(s), start);
                }
                // Ident
                c if c.is_alphabetic() => {
                    while self.next_char_if(char::is_alphabetic).is_some() {}
                    self.end(UasmToken::Ident, start);
                }
                // Number
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

struct Parser<'a> {
    input: &'a str,
    curr: usize,
    tokens: Vec<Sp<UasmToken>>,
    errors: Vec<Sp<LexError>>,
}

impl<'a> Parser<'a> {
    fn next_map<T>(&mut self, f: impl FnOnce(&UasmToken) -> Option<T>) -> Option<Sp<T>> {
        let token = self.tokens.get(self.curr)?;
        let value = f(&token.value)?;
        self.curr += 1;
        Some(token.span.clone().sp(value))
    }
    fn ident(&mut self) -> Option<Sp<&'a str>> {
        let token = self.tokens.get(self.curr)?;
        if !matches!(token.value, UasmToken::Ident) {
            return None;
        }
        self.curr += 1;
        Some(token.span.clone().sp(&self.input[token.span.byte_range()]))
    }
    fn uint(&mut self) -> Option<Sp<u64>> {
        let token = self.tokens.get(self.curr)?;
        if !matches!(token.value, UasmToken::Number) {
            return None;
        }
        let n = self.input[token.span.byte_range()].parse().ok()?;
        self.curr += 1;
        Some(token.span.clone().sp(n))
    }
    fn number(&mut self) -> Option<Sp<f64>> {
        let token = self.tokens.get(self.curr)?;
        if !matches!(token.value, UasmToken::Number) {
            return None;
        }
        self.curr += 1;
        let n = self.input[token.span.byte_range()].parse().unwrap();
        Some(token.span.clone().sp(n))
    }
    fn instr(&mut self) -> Option<Sp<Instr>> {
        let ident = self.ident()?;
        Some(ident.span.sp(match ident.value {
            _ => return None,
        }))
    }
}

pub fn parse(input: &str, path: &Path) -> Result<Assembly, Vec<Sp<ParseError>>> {
    let lexer = Lexer {
        chars: input.chars().collect(),
        loc: Loc::default(),
        src: path.into(),
        tokens: Vec::new(),
    };
    let tokens = lexer.run().map_err(|e| vec![e.map(ParseError::Lex)])?;
    let mut parser = Parser {
        input,
        curr: 0,
        tokens,
        errors: Vec::new(),
    };
    todo!()
}
