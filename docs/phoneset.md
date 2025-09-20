# Somali Phoneset

This document defines the phoneme inventory used in this project and the
mapping between Somali orthographic forms and phonemic symbols.  Somali is a
Cushitic language written in the Latin alphabet.  Its phonological
inventory includes five short vowels and their long counterparts
(`a/aa`, `e/ee`, `i/ii`, `o/oo`, `u/uu`)【275631273323948†L200-L208】, a series of
voiced and voiceless consonants, and a rich system of consonant gemination
where many stops and nasals can be long【275631273323948†L288-L292】.  The phoneset
below extends the ASCII symbols used by eSpeak NG and PL‑BERT to cover
special Somali sounds such as the voiced pharyngeal fricative `ʕ`, the
voiceless pharyngeal fricative `ħ`, and the uvular stop `q`.

## Vowels

| Orthography | Phoneme | Description |
|---|---|---|
| `a` | `a` | open front unrounded vowel |
| `aa` | `aː` | long `a`; vowel length contrasts with `a`【275631273323948†L200-L208】 |
| `e` | `e` | mid front unrounded |
| `ee` | `eː` | long `e` |
| `i` | `i` | close front unrounded |
| `ii` | `iː` | long `i` |
| `o` | `o` | mid back rounded |
| `oo` | `oː` | long `o` |
| `u` | `u` | close back rounded |
| `uu` | `uː` | long `u` |

Long vowels are treated as single phoneme tokens in the training data.  When
phonemizing with eSpeak NG, long vowels are produced as sequences of two
identical vowel symbols; we normalise them to the lengthened forms above.

## Consonants

| Orthography | Phoneme | Notes |
|---|---|---|
| `b` | `b` | voiced bilabial stop |
| `t` | `t` | voiceless alveolar stop |
| `j` | `dʒ` | voiced postalveolar affricate |
| `x` | `ħ` | voiceless pharyngeal fricative【275631273323948†L200-L208】 |
| `kh` | `x` | voiceless uvular fricative |
| `d` | `d` | voiced alveolar stop |
| `dh` | `ð` | voiced dental fricative |
| `r` | `r` | alveolar trill or tap |
| `s` | `s` | voiceless alveolar fricative |
| `sh` | `ʃ` | voiceless postalveolar fricative |
| `c` | `ʕ` | voiced pharyngeal fricative【275631273323948†L200-L208】 |
| `g` | `ɡ` | voiced velar stop |
| `q` | `q` | voiceless uvular stop |
| `k` | `k` | voiceless velar stop |
| `l` | `l` | alveolar lateral approximant |
| `m` | `m` | bilabial nasal |
| `n` | `n` | alveolar nasal |
| `p` | `p` | voiceless bilabial stop (appears only in loanwords) |
| `f` | `f` | voiceless labiodental fricative |
| `w` | `w` | voiced labio‑velar approximant |
| `y` | `j` | palatal approximant |

### Gemination

Somali contrasts singleton and geminate consonants for most plosives and
nasals【275631273323948†L288-L292】.  In our phoneset, a geminate is represented by
doubling the corresponding phoneme.  For example, orthographic `dd`
produces `d d` in the phoneme sequence, and long vowels such as `aa` are
represented as the lengthened monophthong (`aː`).

### Word separator

Following Papercup’s PL‑BERT design, an underscore `_` token is inserted
between words in both the phoneme and grapheme sequences.  This token
functions as a word boundary marker and helps the model learn cross‑word
contexts.

## Notes on eSpeak NG

eSpeak NG provides an IPA transcription mode (`--ipa`) and a mnemonic
phoneme mode (`-x`)【371023369836432†L88-L104】.  When a Somali voice (`so`) is
available, we invoke eSpeak NG with `-v so -x` to obtain mnemonic
phonemes.  Otherwise, the fallback rule‑based phonemizer described above
is used.