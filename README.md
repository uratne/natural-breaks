# natural_breaks

A Rust implementation of the **Jenks natural breaks classification** algorithm
for optimal partitioning of one-dimensional data into `k` classes that minimise
within-class variance (WCSS).

## Features

- Generic over any numeric type that implements `ToPrimitive` (`f64`, `f32`, `i32`, `u64`, etc.)
- Returns clustered values **or** zero-copy index ranges
- Built-in sort variants for unsorted input (with NaN detection)
- Current implementation: **O(kn²)** with an early-exit pruning optimisation

## Quick start

Add to your `Cargo.toml`:

```toml
[dependencies]
natural_breaks = "0.1"
```

### Classify unsorted data

```rust
use natural_breaks::classify_with_sort;

let data = vec![12.0, 1.0, 11.0, 2.0, 10.0, 3.0];
let clusters = classify_with_sort(data, 2).unwrap();
// [[1.0, 2.0, 3.0], [10.0, 11.0, 12.0]]
```

### Classify pre-sorted data

```rust
use natural_breaks::classify;

let data = vec![1, 2, 3, 10, 11, 12];
let clusters = classify(data, 2).unwrap();
// [[1, 2, 3], [10, 11, 12]]
```

### Get index ranges instead of values

```rust
use natural_breaks::classify_indices;

let data = [1.0, 2.0, 3.0, 10.0, 11.0, 12.0];
let ranges = classify_indices(&data, 2).unwrap();
// [(0, 3), (3, 6)]  — half-open ranges [start, end)
```

## API overview

| Function | Input | Returns | Expects sorted? |
|---|---|---|---|
| `classify` | `Vec<T>` | `ClassifiedResult<T>` | Yes |
| `classify_with_sort` | `Vec<T>` | `ClassifiedResult<T>` | No |
| `classify_indices` | `&[T]` | `IndexRanges` | Yes |
| `classify_indices_with_sort` | `Vec<T>` | `IndexRanges` | No |

For direct access to a specific algorithm implementation, use the module:

```rust
use natural_breaks::k_n2::KNSquared;

let clusters = KNSquared::classify(vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0], 2).unwrap();
```

## Algorithm

Based on:

> Wang & Song, "Optimal Classification of Quantitative Data",
> *The R Journal*, Vol. 3/2, December 2011.
> https://journal.r-project.org/articles/RJ-2011-015/

This implementation adds an early-exit pruning step that breaks the inner loop
when the running within-cluster sum of squares already exceeds the current best,
exploiting the monotonicity of WCSS on sorted data.

## Roadmap

### O(kn log n)

Next up is an implementation of Fisher's natural breaks with **O(kn log n)**
complexity. Will be implemented soon. Based on:

> [Fisher's Natural Breaks Classification — Complexity Proof](https://geodms.nl/docs/fisher%27s-natural-breaks-classification-complexity-proof.html)

### O(kn)

After that, I plan to explore an **O(kn)** algorithm based on:

> Xiaolin Song *et al.*, "An optimal-time algorithm for the k-filling problem and its application to the one-dimensional Jenks classification",
> *Bioinformatics*, Vol. 36, Issue 20, October 2020.
> https://academic.oup.com/bioinformatics/article/36/20/5027/5866975

This one still needs investigation and may take some time.

## Learning resources

If you are new to natural breaks / Jenks classification, this is a good starting
point:

> [EHDP — Jenks Natural Breaks](https://www.ehdp.com/methods/jenks-natural-breaks-2.htm)

## License

See [LICENSE](LICENSE) for details.