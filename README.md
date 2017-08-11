# Modified Norvig

in 2007, Peter Norvig shared what has probably become one of the simplest and broadly sited spelling correction tool written in python, which is [described in detail on his website](http://norvig.com/spell-correct.html).

His code is used in other nlp tools, presumably because it is rock solid and super fast. The tool relies solely on ranking correction candidates based on the frequency that they show up in a lexicon. Most of the time, this is totally valid metric, but in cases where you only have a list of words without frequencies or, if your correcting for human typing errors, frequency along may fall short.

With the provided code, I have implemented Norvig's spelling corrector, but have also added a keyboard distance modification. This slows the process down a little, but does provide a distance metric one can use for doing things like fuzzy matching using a model lexicon loaded into memory, which can speed up NLP pipelines doing weak supervision/distance learning, for example.

## Requirements

- a raw text file or delimited word frequency file with all the words you are looking to correct for.
- Python3

## How it works

```python
>>> from code import spelcor
>>> sp = spelcor(wordsfile='data/big.txt',lower=True)
>>> sp.cor_norvig('dpellimg')
    'dwelling'
>>> sp.cor_modNorvig('dpellimg')
    'spelling'
>>> sp.candidates
    [['spelling', 0.091969860292860584, 5],
     ['dwelling', 0.084172191121783826, 7]]
```

For the standard Norvig correction return is `dwelling` because it occurs with the highest frequency in provided words file.

For the modified Norvig the return is `spelling`, which occurs 2 fewer time in the word file than `dwelling`, but is ranked higher with respect to keyboard distance.
