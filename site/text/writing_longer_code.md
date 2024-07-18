# Writing Longer Code

So far, most of the code examples in this tutorial have been fairly short. While Uiua is great for short, simple code, it is designed to be a general-purpose language. Uiua aims to be decent for everything from Code Golf to command-line utilities, from websites to games.

However, it may not be immediately clear how to write more complex code that does not refer to variable names.

## What Uiua asks of you

When you first start using Uiua beyond just simple examples and challenges, you will likely encounter difficulty passing more than a couple values around.

In disallowing named local variables, Uiua asks something of the programmer that most other languages do not. It asks that you to re-orient the way you think about data. How you refer to it, how it flows through a program.

If you pay the price of this re-orientation, Uiua offers you a few things in return.

For one, you eliminate the need to name things that are not worth naming. Consider the classic C `for` loop `for (int i = 0; i < len; i++)`. Why do we have to write all that for such a fundamental concept? Why do we call it `i`? In more modern languages like Rust, we have range-based for loops. But even in `for item in items`, we still have to come up with a name for `item`, even though this name is irrelevant to actually solving our problem. These are simple examples of the classic problem in computer science of naming things.

Other array languages like APL, J, and BQN also allow you to write highly *tacit* code. Even some non-array languages like Haskell allow a high degree of tacitness.

Uiua goes a step further by requiring *all* code to be tacit. In doing this, we eliminate the *ceremony* of naming.

Ceremony in programming is the bits of code you have to write not because they are part of solving your problem, but because they are part of the syntax or semantics of the language.

Different languages require different amounts of ceremony. Uiua tries to eliminate it as much as possible while remaining readable.

## The Stack Pitfall

Being stack-based is Uiua's key to being usable as a pure-tacit language. However the stack can be an unwieldy tool if used recklessly. Many stack languages have built-in functions for rotating the stack, fishing values from deep in the stack, or arbitrarily reordering it. While these things are technically possible in Uiua, they are discouraged, and they are verbose by design.

Uiua encourages a more structured approach to stack manipulation. There are no single functions for rotating the stack or for swapping more than 2 values.

When complex stack manipulation *is* required, it is usually done with [planet notation](/tutorial/advancedstack#planet-notation). Planet notation allows you to *visualize* the way values move around.

## A Motivating Example

The online Uiua pad and the `uiua watch` command in the native interpreter make it easy to write Uiua code interactively. You can easily see the state of the stack after each change you make to the code.

Unfortunately, while this iterative process is good for exploring possibilities, an ad-hoc approach to stack manipulation often leads to code that is very hard to read.

As a motivating example, let's attempt to implement the quadratic formula. Given numbers `a`, `b`, and `c`, the roots of the function `ax² + bx + c` can be found via the expression `(-b ± √(b² - 4ac)) / 2a`.

This is a useful example because it involves juggling 3 arguments that are used in a non-regular way.

Let's start with the discriminant term ` √(b² - 4ac)`.

```uiua

```