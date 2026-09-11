<!-- Lifted from tig-foundation/prometheus-swarm scripts/prompts.py (GPLv3) -->

EXTRA RULES FOR A CLEAN COMPILE (smaller models — follow ALL of these):

RULE 1 - NO DUPLICATE STRUCTS:
  Each struct (e.g. a Hyperparameters config) is defined ONCE. Modify the
  existing definition in place; never add a second
  `pub struct <Name> { ... }` with a name that already exists.

RULE 2 - BORROW CHECKER:
  Copy a value out before mutating the collection it came from.
  BAD:  let item = vec.choose(&mut rng).unwrap(); vec.retain(...);
  GOOD: let item = *vec.choose(&mut rng).unwrap(); vec.retain(...);
  Alternative: remove by index with `let x = vec.remove(idx);`. When the
  borrow checker objects, clone the data rather than leaving code that won't
  build.

RULE 3 - TRAIT IMPORTS:
  Import every trait whose methods you call, with the other `use` lines at the
  top of the file — e.g. `use rand::prelude::{SliceRandom, IteratorRandom};`
  for `.choose()` / `.shuffle()`. A missing trait import is a compile error.

RULE 4 - BRACE BALANCE:
  Before returning, verify every `{`, `(`, and `[` has a matching close.

RULE 5 - COUNT SYMMETRIC PAIRS ONCE:
  When summing over a symmetric matrix, iterate unordered pairs only:
  `for i in 0..n { for j in (i+1)..n { /* use m[i][j] */ } }` — double-counting
  silently doubles the objective.

RULE 6 - USE i64 FOR ACCUMULATORS:
  Sum into an i64 to avoid u32/i32 overflow and to allow negative deltas:
  `let mut total: i64 = 0; total += value as i64;`

RULE 7 - DEFINE BEFORE USE:
  Every variable must be declared before its first use within its scope; don't
  reference a binding from a sibling or inner block that isn't visible there.

GENERAL COMPILE HYGIENE:
- Before finishing, mentally run `cargo check`: every variable is used or
  prefixed with `_`; every `match` is exhaustive; every branch returns the
  same type; no semicolon dropped where a value is expected.
- Prefer iterator methods you are sure of (`.iter()`, `.enumerate()`,
  `.map()`, `.filter()`, `.sum()`, `.min()/.max()`) over hand-written index
  loops; when you do index, derive the bound from `.len()`.
- Annotate numeric literals when the type is ambiguous (`0usize`, `1.0f64`).
  Use `as f64` / `as usize` for casts; never rely on implicit coercion. Calling
  a method on a bare literal needs the type pinned: write `2.0_f64.sqrt()` or
  `(n as f64).powi(2)`, never `2.0.sqrt()` (that is `E0689: can't call method on
  ambiguous numeric type`).
- Reading config from parsed JSON: a `Map<String, Value>` is NOT a `Value`, so
  do NOT `serde_json::from_value(...)` the whole map (that is a type error) —
  pull keys individually with a default. `Value::as_f64()` returns an `f64`, so
  the `.unwrap_or(...)` argument must be an `f64` literal and you cast to `f32`
  only AFTER `unwrap_or`: `m.get("lr").and_then(|v| v.as_f64()).unwrap_or(0.001)
  as f32` — passing an `f32` to `.unwrap_or` here is `expected f64, found f32`.
- Declare `let mut x = ...` whenever you later mutate `x` or call a `&mut`
  method on it; "cannot borrow as mutable, not declared as mutable" just means a
  missing `mut`.
- Any struct YOU define that you `.clone()`, sort, or push into a `BinaryHeap`
  must carry the right derives: `#[derive(Clone)]` (and additionally
  `#[derive(PartialEq, Eq, PartialOrd, Ord)]` for a `BinaryHeap`). Without
  `#[derive(Clone)]`, `x.clone()` silently clones a `&reference` instead and the
  next mutation fails with `E0596: cannot borrow ... as mutable`.
- A trait method only works if the trait is in scope. If rustc says `method not
  found ... trait X ... is implemented but not in scope`, add the `use` it
  suggests — do NOT rewrite the call into something else.
- Don't introduce new generics, trait bounds, lifetimes, or macros unless the
  starting code already uses them — they are a common source of errors.
- Reuse the data structures already imported at the top of the file; don't invent
  types, constants, or functions that aren't defined. A `crate::...::NAME` path
  (or a bare name) that isn't actually declared is `E0425: cannot find
  value/function`. If you need a threshold or hyperparameter, define it as a
  local `const` in your own file (e.g. `const MIN_LOSS_DELTA: f32 = 1e-4;`)
  rather than referencing one you assume the module exports.
- Indexing a `Vec`/slice MOVES the element when it isn't `Copy` — that's
  `E0507: cannot move out of ... behind a shared reference`. Bind by reference
  (`let x = &v[i];`) or `.clone()` it; never `*v[i]` or destructure-by-value out
  of a borrowed container (e.g. `let (_, s) = *states[k].iter().max()...;`).
- Prefer small, incremental edits over a full-file rewrite: most compile
  failures come from rewriting the whole file and letting signatures or imports
  drift. Change the algorithm bodies and leave the boilerplate intact.
- Keep the change focused: modify the algorithm logic, not the function
  surface, so the result still slots into the existing module.