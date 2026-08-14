Yes. There are several things I found that are **not yet proven to be wrong from the two Python files alone**, because they depend on what is actually stored inside the `.pt` shards. Those are exactly the things I would investigate next rather than “fix” blindly.

Here is the **dataset audit checklist** I would use. This is deliberately about **verifying what the dataset contains and whether the training-side interpretation matches it**, not changing the preprocessing.

# 🐉 Dataset Audit TODO

## A. Gloss token-space audit

### A1. Determine what `rec["gloss_seq"]` actually contains

The indexer has two possible paths:

```python
if "gloss_seq" in rec:
    token_ids = gs.tolist()
```

while later the dataset does:

```python
raw_gloss_seq = [BOS_ID] + [
    t + GLOSS_OFFSET if t >= 0 else UNK_ID
    for t in token_ids
] + [EOS_ID]
```

 

### Investigate

For several actual records containing `gloss_seq`, determine:

```text
Does gloss_seq contain:
[raw class IDs]
[IDs already offset by +4]
[BOS/EOS]
[PADDING]
[some combination]
```

### Test with examples

For one known sign whose global raw class ID is `X`, inspect:

```text
stored gloss_seq
stored label_idx
vocabulary mapping
generated dataset["gloss_seq"]
```

You want to prove the exact transformation:

```text
stored ID
   ↓
indexer token_ids
   ↓
+ GLOSS_OFFSET
   ↓
model target
```

### Red flag

Anything like:

```text
stored = 17
model target = 21
```

is expected **only if stored 17 is a raw class ID**.

But:

```text
stored = 21
model target = 25
```

would indicate double offsetting if 21 was already a model token ID.

---

# B. Special-token audit

For every stored sequence field:

```text
gloss_seq
chicago_seq
english_seq
```

inspect whether the shard already contains:

```text
PAD = 0
BOS = 1
EOS = 2
UNK = 3
```

or whether those are introduced only by the training dataset.

The current dataset code initializes:

```python
raw_gloss_seq = [BOS_ID, EOS_ID]
raw_chicago_seq = [BOS_ID, EOS_ID]
raw_english_seq = [BOS_ID, EOS_ID]
```

and then constructs the sequence itself. 

### Investigate

Find 20-50 records and compare:

```text
stored length
stored first token
stored last token
stored unique IDs
```

against:

```text
expected raw label/gloss sequence
```

### Red flags

Any stored sequence already containing:

```text
BOS / EOS / PAD
```

when the loader assumes raw class IDs.

Also check whether PAD is ever a legitimate raw class ID.

---

# C. Label-space consistency audit

You have at least these representations:

```text
label_idx
token_ids
gloss_seq
class_counts
label_to_idx
GlossVocabulary
```

The dataset converts among these several times. Because humans apparently decided one ID space wasn't sufficiently entertaining.

### Investigate

For every sampled record construct a table:

| Field                 | Example |
| --------------------- | ------: |
| `raw_label`           |     ... |
| `label_idx`           |     ... |
| `token_ids`           |     ... |
| generated `gloss_seq` |     ... |
| vocabulary ID         |     ... |
| model target ID       |     ... |

Then verify:

```text
label_idx == label_to_idx[raw_label]
```

and for isolated samples:

```text
token_ids == [label_idx]
generated target == [BOS, label_idx + 4, EOS]
```

### Red flags

Any of:

```text
label_idx != mapping[label]
negative token ID
token ID >= number of real classes
token ID already includes +4
different IDs for identical gloss strings
```

---

# D. Vocabulary contiguity audit

The loader assumes:

```python
lbl_idx >= len(self.label_to_idx)
```

means invalid. 

That only works if IDs are contiguous.

### Investigate

Extract every value from:

```text
label_to_idx
```

and calculate:

```text
min ID
max ID
number of unique IDs
number of mappings
missing IDs
duplicate IDs
```

You want:

```text
min = 0
max = N-1
unique count = N
```

### Red flags

Something like:

```text
IDs: 0, 1, 2, 4, 5
```

because then ID `3` is missing and `len(mapping)=5` does not describe the actual ID range correctly.

This isn't proven to be wrong in your dataset. It's something the dataset audit should establish.

---

# E. Cross-source label collision audit

This one is particularly important in Frankenstein-style mixtures.

Your loader routes records by:

```python
task_str
source_str
raw_label
```

with different target spaces for:

```text
isolated gloss
Chicago fingerspelling
How2Sign English
fallback gloss
```



### Investigate

For each `(source, task)` combination, collect:

```text
label string
label_idx
token_ids
target vocabulary
```

Then check whether the **same integer ID** is being interpreted as different semantic things across tasks.

For example:

```text
ID 17 → ASL gloss "BOOK"
ID 17 → some unrelated source-specific label
```

That's acceptable only if those IDs never share a head.

But because the generic `label` field feeds multiple auxiliary systems, this deserves explicit verification.

---

# F. `label_idx` semantics for non-isolated records

The loader sets:

```python
lbl_clean = -1
```

for Chicago and sentence-level records during indexing. 

But later `__getitem__` returns:

```python
"label": label_idx
```

and the training loop still uses `labels` for things like auxiliary and contrastive supervision. 

### Investigate

For every Chicago / How2Sign record:

```text
label
is_isolated
task
source
has_valid_gloss
has_valid_chicago
has_valid_english
```

Verify that:

```text
non-isolated sample → label = -1
```

or whatever the intended sentinel is.

### Red flag

Any sentence/fingerspelling record accidentally retaining an ordinary ASL class ID.

That could leak an isolated-class objective into the wrong sample.

---

# G. `is_isolated` audit

The loader decides:

```python
is_isolated = True
```

for:

```text
isolated_gloss
static_alphabet
isolated_number
```

and in the fallback branch:

```python
is_isolated = len(token_ids) <= 1
```



### Investigate

For a large random sample, compare:

```text
task_str
token_ids length
is_isolated
label_idx
source
```

### Red flags

Examples:

```text
sentence with one gloss → is_isolated=True
isolated sample with multiple token_ids → is_isolated=True
unknown task with one token → is_isolated=True
```

The fallback rule is especially worth auditing because it makes a semantic assumption solely from token count.

---

# H. Sequence truncation audit

We already established the loader records truncation:

```python
gloss_trunc
chicago_trunc
english_trunc
```

but still trains on those sequences. 

Now audit the **actual frequency and shape**.

### Compute

For each split:

```text
number of gloss sequences > 64
number of Chicago sequences > 64
number of English sequences > 128
percentage truncated
distribution of original lengths
distribution of truncated lengths
```

### Crucially

For truncated examples, inspect:

```text
last original token
last stored target token
whether EOS was originally present
whether loader replaced it
```

You already know the loader will fabricate EOS at the boundary. 

The unresolved dataset question is:

> **How many real samples are affected?**

---

# I. Empty-sequence audit

The loader initializes every target to:

```python
[BOS, EOS]
```

and marks validity separately. 

### Investigate

Count records where:

```text
has_valid_gloss=False
has_valid_chicago=False
has_valid_english=False
```

Also count:

```text
has_valid_X=True
sequence has zero actual lexical tokens
```

### Red flag

A sample being marked valid while its target is effectively:

```text
BOS EOS
```

because that becomes a legitimate termination target despite carrying no actual content.

---

# J. Unknown-label audit

The skip list is:

```python
{
    "",
    "unknown",
    "none",
}
```



### Investigate

Search actual shard metadata for variants such as:

```text
<unk>
UNK
unknown_sign
Unknown
N/A
null
NULL
?
undefined
background
no_label
```

and determine whether any are being treated as genuine labels.

### Red flag

A placeholder label that is not included in `_SKIP_LABELS`, therefore becoming a real training target.

This needs dataset inspection because the loader alone cannot tell us what strings actually exist.

---

# K. Sample-weight audit

This deserves a **dataset-side** investigation separate from the training normalization bug.

The loader obtains:

```python
float(rec.get("quality", rec.get("sample_weight", 1.0)))
```



### Investigate

Distribution of:

```text
sample_weight
quality
```

per source/task.

Calculate:

```text
min
max
mean
median
p1/p99
zero count
negative count
NaN count
Inf count
```

### Red flags

Especially:

```text
weight = 0
weight < 0
weight >> 1
NaN
Inf
```

A negative sample weight would literally reverse the gradient contribution for weighted losses.

The current loader doesn't validate it.

---

# L. Quality-weight correlation audit

This is a more subtle dataset audit.

For every source:

```text
sample_weight distribution
sequence length
source
task
label frequency
```

### Investigate

Ask:

> Are weights correlated with particular classes, datasets, or sequence lengths?

For example:

```text
How2Sign → weight 0.5
isolated Citizen → weight 2.0
Chicago → weight 0.8
```

That may be intentional.

But if one source systematically receives higher weights, your nominal dataset mixture is not your **effective** training mixture.

This becomes important because the losses already apply different task masks.

---

# M. Class-count audit

The dataset counts tokens while indexing:

```python
for t in token_ids:
    if t >= 0:
        local_counts[int(t)] += 1
```



### Investigate

Compare:

```text
class_counts
```

against actual **training occurrences** after:

```text
truncation
task routing
has_valid masks
sample filtering
augmentation
```

The current counts are collected before those later transformations.

### Red flag

If `class_counts` is supposed to represent training-token frequency but includes records/tokens that never reach the actual gloss loss.

That would make class weighting based on the wrong population.

---

# N. Duplicate-record audit

This is one I would absolutely do with a 200k-clip Frankenstein dataset.

For every record, derive something like:

```text
hash(feature content)
hash(label metadata)
source
task
label
```

Then identify:

```text
exact feature duplicates
same feature + different label
same underlying clip appearing in train and val
same clip under different source names
```

### Most important test

Check:

```text
train feature hash ∩ val feature hash
```

A nonempty intersection is a potentially catastrophic split leak.

I'm **not claiming you have leakage**. The loader doesn't establish that either way.

This is simply one of the highest-value things to audit.

---

# O. Near-duplicate audit

Exact hashes aren't enough.

For landmark sequences, look for:

```text
same source video
same signer
same temporal window
same normalized landmark trajectory
```

appearing in multiple splits.

Particularly inspect:

```text
How2Sign
ASL Citizen
ChicagoFSWild
```

because different annotations or generated clips can potentially originate from overlapping source material.

Again, this requires inspecting dataset metadata. The Python loader cannot prove it.

---

# P. Source/task balance audit per TPU shard

Because the dataset is sharded and workers are assigned subsets, calculate for **each shard**:

```text
% isolated
% Chicago
% How2Sign
% other
% valid_gloss
% valid_chicago
% valid_english
```

Then calculate this again **per TPU worker's assigned shard set**.

### Red flag

Something like:

```text
TPU 0:
80% isolated

TPU 1:
60% How2Sign

TPU 2:
95% Chicago

TPU 3:
70% isolated
```

That interacts badly with the local-normalization issue we already found.

---

# Q. Actual sequence-length semantics audit

The loader's `gloss_len` is:

```python
actual_len = min(len(raw_seq), max_len)
```



And because `raw_seq` includes BOS/EOS, `gloss_len` includes those special tokens.

### Investigate

Determine what every downstream consumer thinks `gloss_len` means:

```text
total target length?
number of gloss tokens?
number of decoder outputs?
CTC target length?
```

The CTC implementation does **not** use `gloss_len` directly to determine its target length. It recomputes:

```python
tgt_lengths = valid_mask.sum(...)
```



### Red flag

If some other training/metrics code assumes:

```text
gloss_len = number of lexical glosses
```

when it actually means:

```text
BOS + lexical tokens + EOS
```

That would produce systematic off-by-two errors.

---

# R. Chicago vocabulary audit

The loader documents:

```text
PAD=0
BOS=1
EOS=2
UNK=3
SP=4
a-z=5-30
0-9=31-40
```

and the decoder is constructed with:

```python
vocab_size=42
```

 

### Investigate

Actually enumerate every token produced by the loader and verify:

```text
min >= 0
max < 42
SP used correctly
digits occupy 31-40
EOS always 2
UNK only appears for intended characters
```

Also inspect real labels containing:

```text
apostrophes
hyphens
punctuation
uppercase
unicode characters
```

because the tokenizer converts anything outside the explicitly supported characters to `UNK`.

That's a **dataset-content audit**, not a preprocessing critique.

---

# S. English vocabulary audit

Your encoder does:

```python
text.strip().lower().split()
```

and maps unknown words to `<UNK>`. 

### Investigate actual English labels for:

```text
punctuation
contractions
possessives
hyphenated words
numbers
apostrophes
Unicode
```

Then calculate:

```text
% tokens mapped to UNK
% sequences containing ≥1 UNK
UNK rate by source
UNK rate by word frequency
```

The English vocabulary itself may be perfectly constructed, while the actual dataset contains text the tokenizer cannot represent.

That's exactly what you want the audit to reveal.

---

# T. ASL-LEX coverage audit

The lexical embedding table only populates rows when it finds a matching vocabulary entry in the CSV. 

### Calculate

```text
total gloss classes
classes with ASL-LEX metadata
classes without metadata
% coverage
```

Then:

```text
coverage by source
coverage by frequency
coverage of top 100 classes
```

### Red flag

If 95% of your **rare classes** have no ASL-LEX metadata while common classes do, the lexical auxiliary isn't just incomplete. Its incompleteness is class-frequency-dependent.

---

# U. ASL-LEX attribute collision audit

The mappings are generated from whatever order the CSV rows are encountered in:

```python
lexclass_map[lc] = len(lexclass_map)
```

and similarly for the other attribute categories. 

### Investigate

Dump:

```text
attribute string → assigned integer
```

for:

```text
LexicalClass
SignType
Handshape
MajorLocation
SemanticCategory
```

and verify that the same string always gets the same ID across training runs / files.

This is particularly worth checking if different preprocessing versions used different CSVs.

---

# V. Frame-index audit after temporal modification

The loader preserves:

```python
frame_indices
```

and later reconstructs kinematics from them. 

### Investigate actual samples after downsampling/augmentation

Check invariants:

```text
frame_indices are monotonically increasing
frame_indices are unique
frame_indices correspond to actual retained frames
frame_indices length == feature length
padded frame indices don't masquerade as real indices
```

Especially check:

```text
mask[i] == False
```

positions.

You don't need to change anything yet. Just establish whether:

```text
invalid/padded position
frame_index = 0
```

can be confused with a legitimate first frame.

---

# W. Zero-frame / malformed-feature audit

`__getitem__` explicitly supports:

```python
feat_arr.ndim != 2 and feat_arr.ndim != 3
    → T = 0
```



And then returns an all-zero sequence with an all-false mask.

### Investigate

Count:

```text
T == 0
T == 1
T < minimum useful sequence length
NaN-containing records
Inf-containing records
unexpected dimensions
```

Because these records aren't necessarily rejected. They can become samples containing no valid frames.

---

# X. Feature-dimension audit

The loader silently does:

```text
D > expected → truncate
D < expected → zero-pad
```



### Investigate

Before accepting this behavior as intended, measure:

```text
feature dimension distribution
dimensions by source
dimensions by task
```

and identify every distinct value.

### Red flag

Multiple feature schemas existing inside the same training dataset.

For example:

```text
source A → 540 dims
source B → 507 dims
source C → 420 dims
```

Then the loader creates a common tensor by truncation/padding, which might mean the model is receiving different semantics in the same channel positions.

Again, **do not fix this yet**, just establish whether it actually occurs.

---

# The audit order I'd use

Don't brute-force all 20 things immediately. I'd attack them in this order:

```text
1. Stored gloss_seq token space
2. Stored special tokens
3. label_idx ↔ vocabulary mapping
4. Train/val exact duplicates
5. Cross-source label collisions
6. is_isolated correctness
7. Truncation frequency + actual target mutation
8. Empty/invalid targets
9. sample_weight distribution
10. per-shard task distribution
11. class_counts vs actual loss population
12. English UNK rate
13. ASL-LEX coverage
14. frame-index invariants
15. malformed/zero-frame records
16. feature-schema distribution
17. Chicago token distribution
18. vocabulary contiguity
19. near-duplicate split leakage
20. TPU-worker distribution
```

### The four I most want you to investigate first

**#1 `gloss_seq` ID-space semantics**

This has the possibility of causing **systematic label corruption** if the stored sequence is already offset.

**#4 train/validation duplication**

With ~200k clips, a tiny accidental overlap can produce absurdly good validation results without the model actually generalizing.

**#7 truncation prevalence**

We already proved the loader mutates truncated targets. The open question is how much of your dataset is actually affected.

**#10 per-shard task balance**

This determines how badly the distributed-loss normalization bug actually distorts the training objective.

Those four can tell us whether we're looking at isolated implementation bugs or a deeper dataset/training-contract mismatch. 🐲


at E:\datasets\asl_dataset\asl_preprocessed_phase1
remember the machine you are using to audit only has 16gb of ram (like 10gb free after other services), so dont open too many other programs
this is a rather large dataset, so be patient and let the code run, it will crash antigravity and your workflow.
running on a i5-8250u