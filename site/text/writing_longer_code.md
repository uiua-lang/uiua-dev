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
Disc ←
Disc 1 2 0
```

To show how you might build up a solution with only stack reordering, we'll only use [`duplicate`](), [`flip`](), [`over`](), and [`dip`]() to attempt to get all the arguments in the right order.

First, we'll might try to get `a` and `c` next to each other above `b` on the stack.

```uiua
Disc ← ⊙:
Disc 1 2 0
```

Then, we can create the `4ac` and `b²` terms and [subtract]().

```uiua
Disc ← -××4⊙⊙(ⁿ2) ⊙:
Disc 1 2 0
```

Then we'll account for [complex]() roots and take the [sqrt]().

```uiua
Disc ← √ℂ0 -××4⊙⊙(ⁿ2) ⊙:
Disc 1 2 0
```

That finishes the discriminant.
We can implement `±` by [couple]()ing the value with it's [negate]().

```uiua
Quad ← ⊟¯. √ℂ0 -××4⊙⊙(ⁿ2) ⊙:
Quad 1 2 0
```

And now we have a problem. We still need to use `a` and `b` one more time, but they have already been consumed.
`a` and `b` start at the top of the stack, so we can copy them with [over]() and put the rest of out code in two [dip]()s.

```uiua
Quad ← ⊙⊙(⊟¯. √ℂ0 -××4⊙⊙(ⁿ2) ⊙:),,
Quad 1 2 0
```

Then we'll [subtract]() `b`... 

```uiua
Quad ← ⊙(-⊙(⊟¯. √ℂ0 -××4⊙⊙(ⁿ2) ⊙:)),,
Quad 1 2 0
```

...and [divide]() by `2a`.

```uiua
Quad ← ÷×2⊙(-⊙(⊟¯. √ℂ0 -××4⊙⊙(ⁿ2) ⊙:)),,
Quad 1 2 0
```

And their we have it, the quadratic formula.

```uiua
Quad ← ÷×2⊙(-⊙(⊟¯. √ℂ0 -××4⊙⊙(ⁿ2) ⊙:)),,
Quad 1 2 0
Quad 1 2 5
Quad 2 3 1
```

On close inspection, the astute reader may notice that the above code sucks. What's worse, it's not even as bad as it could be. If you hadn't thought to use [over]() and [dip]() in that way, you may have instead used the dreaded `:⊙:` to rotate 3 values on the stack.

The problem with reordering stack values this often is that the state of the stack at any point in the code gets harder and harder for the writer to keep in their head. It also makes it much harder for the reader to deduce the state of the stack at a glance.

## Stack-Source Locality

The code above is also obtuse for another reason.

Imagine a person who is less familiar with this code going to read it. It may be someone else, but it may also be a future version of yourself. If they look at the leftmost term `÷×2`, they'll likely be able to quickly tell that it takes two arguments. But how do they figure out what those arguments are? They would have to make their way all the way to the *other side of the function* to find the [over]() that creates the copy of `a`. They would only end up there after having built up the mental model of the state of the stack throughout the *entire function*.

This obtuseness is the result of the above code violating a fundamental principal of writing good Uiua code, that of *stack-source locality*. Stated simply, **code that creates values should be as close as possible to the code that uses those values**.

In our example, [divide]() and [over]() are on opposite sides of the function: a massive violation of stack-source locality.

This principal is not a formula you can plug values into. It is not a set of procedures that will make code better. It is a guiding tenet meant to shape the way you think about the flow of your data and how you structure your programs. How well a given code snippet maintains stack-source locality is up to interpretation, and different Uiua programmers may interpret it differently, even for the same program.

## A Better Way

So how do we write better Uiua code? How do we keep stack-source locality? How do we avoid making the stack so convoluted that our code becomes unreadable.

The short answer is to make liberal use of [fork]().

The power of [fork](), [dip](), [gap](), [on](), and [by]() is that they allow access to arbitrary values on the stack *without* reordering it. When the stack maintains its order, it is much easier to reason about values' position on it, since their positions seldom change relative to each other.

Let's redo the quadratic formula implementation using these modifiers.

We'll start again with the discriminant.

```uiua
Disc ← -⊃(××4⊙⋅∘)⋅(ⁿ2)
Disc 1 2 0
```

Notice that when we use planet notation, it is easier to tell which functions are being applied to which values.

We'll implement the `√` and `±` in the same way as before.

```uiua
Disc ← ⊟¯. √ℂ0- ⊃(××4⊙⋅∘)⋅(ⁿ2)
Disc 1 2 0
```

Even though `b` has been consumed, we can gain access to it again using another [fork]() and implement the `-b` term.

```uiua
Quad ← -⊃⋅∘(⊟¯. √ℂ0 -⊃(××4⊙⋅∘)⋅(ⁿ2))
Quad 1 2 0
```

Then, we can use another [fork]() to add the `/ 2a` part.

```uiua
Quad ← ÷⊃(×2|-⊃⋅∘(⊟¯. √ℂ0 -⊃(××4⊙⋅∘)⋅(ⁿ2)))
Quad 1 2 0
```

Long lines like this can hurt readability. One thing we can do to alleviate this is split the discriminant onto its own line.

```uiua
Quad ← ÷⊃(×2)(
  -⊃⋅∘(
    -⊃(××4⊙⋅∘)⋅(ⁿ2)
    ⊟¯. √ℂ0
  )
)
Quad 1 2 0
```

Alternatively, we can pull the discriminant into its own function.

```uiua
# A thing of beauty
Disc ← -⊃(××4⊙⋅∘)⋅(ⁿ2)
Quad ← ÷⊃(×2|-⊃⋅∘(⊟¯. √ℂ0 Disc))
Quad 1 2 0
```

Let's compare this solution to the previous one. To improve the comparison, we'll make the discriminant its own function here as well.

```uiua
Disc ← -××4⊙⊙(ⁿ2) ⊙:
Quad ← ÷×2⊙(-⊙(⊟¯. √ℂ0 Disc)),,
Quad 1 2 0
```

The difference is night-and-day. The old, naive solution, even with the benefit of being broken up, still has all of its same issues.

If we look in the improved solution and do the same search for the source of [divide]()'s arguments, we don't have to go far before finding the [fork]() with `×2` and `-`. Stack-source locality holds for all parts of the code!