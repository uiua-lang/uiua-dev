# Writing Longer Code

So far, most of the code examples in this tutorial have been fairly short. While Uiua is great for short, simple code, it is designed to be a general-purpose language. Uiua aims to be decent for everything from Code Golf to command-line utilities, from websites to games.

However, it may not be immediately clear how to write more complex code that does not refer to variable names.

## What Uiua asks of you

When you first start using Uiua beyond just simple examples and challenges, you will likely encounter difficulty passing more than a couple values around.

In disallowing named local variables, Uiua asks something of the programmer that most other languages do not. It asks that you to re-orient the way you think about data. How you refer to it, how it flows through a program.

If you pay the price of this re-orientation, Uiua offers you a few things in return.

For one, you eliminate the need to name things that are not worth naming. Consider the classic C `for` loop `for (int i = 0; i < len; i++)`. Why do we have to write all that for such a fundamental concept? Why do we call it `i`? In more modern languages like Rust, we have range-based for loops. But even in `for item in items`, we still have to come up with a name for `item`, even though this name is irrelevant to actually solving our problem. These are simple examples of the classic problem in computer science of naming things.

