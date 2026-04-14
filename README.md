<!-- ████████████████████████████████  HEADER  ████████████████████████████████ -->

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=1,8,15&height=260&section=header&text=ShakeGen&fontSize=72&fontColor=ffffff&fontAlignY=40&desc=Shakespeare%20Text%20Generation%20%E2%80%A2%20GRU%20%E2%80%A2%20Character-Level%20RNN%20%E2%80%A2%20TensorFlow&descAlignY=62&descSize=19&animation=fadeIn&stroke=7C3AED&strokeWidth=1" width="100%"/>

<!-- ████████████████████████████████  TYPING  ████████████████████████████████ -->

<div align="center">

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&duration=3000&pause=800&color=7C3AED&center=true&vCenter=true&multiline=false&repeat=true&width=720&height=50&lines=GRU+%E2%86%92+Shakespearean+Text+%E2%80%94+Character+by+Character+%F0%9F%8E%AD;1.1M+Characters+%C2%B7+128+GRU+Units+%C2%B7+Temperature+Sampling+%E2%9A%A1;60%25+Accuracy+%C2%B7+10+Epochs+%C2%B7+3.8+Hours+of+Training+%F0%9F%A7%A0;AASD-4011+%E2%80%94+Advanced+Math+for+Deep+Learning+%C2%B7+GBC+%F0%9F%8E%93)](https://git.io/typing-svg)

</div>

<br/>

<!-- ████████████████████████████████  BADGES  ████████████████████████████████ -->

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![GRU](https://img.shields.io/badge/Model-GRU_RNN-7C3AED?style=for-the-badge)](.)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Dataset](https://img.shields.io/badge/Dataset-Shakespeare_1.1M_chars-22C55E?style=for-the-badge)](.)
[![Accuracy](https://img.shields.io/badge/Val_Accuracy-~55%25-F59E0B?style=for-the-badge)](.)
[![Course](https://img.shields.io/badge/GBC-AASD--4011_Adv_Math_DL-8B5CF6?style=for-the-badge)](.)

</div>

<br/>

---

<!-- ████████████████████████████████  ABOUT  ████████████████████████████████ -->

## 🧠 Project Overview

```python
class ShakeGen:
    def __init__(self):
        self.authors   = ["Anisha Singla", "Partner"]
        self.course    = "AASD-4011 — Advanced Mathematics for Deep Learning"
        self.college   = "George Brown College · Winter 2023"
        self.task      = "Character-level text generation — fake Shakespearean prose"
        self.dataset   = "Complete works of Shakespeare (~1.1M characters)"
        self.reference = "Aurélien Géron — Hands-On Machine Learning (3rd Ed.)"

    @property
    def architecture(self):
        return {
            "embedding"  : "Embedding(vocab_size, 16) — dense char vectors",
            "recurrent"  : "GRU(128)                 — gated recurrent unit",
            "output"     : "Dense(39, softmax)        — 39 unique characters",
            "window"     : "100 characters            — sliding-window sequences",
            "training"   : "10 epochs · ~3.8 hours   — tensorflow-cpu",
            "result"     : "~60% train acc · ~55% val acc",
        }

    def generate(self, seed: str, temperature: float = 1.0) -> str:
        # Temperature controls creativity vs coherence
        # Low  (0.01) → repetitive but coherent
        # Mid  (1.0)  → balanced
        # High (100)  → chaotic, random
        return autoregressive_sample(seed, temperature)
```

> Feed the complete works of Shakespeare into a GRU network — one character at a time — and watch it learn to hallucinate new Shakespearean prose. A character-level language model before transformers made it look easy.

---

<!-- ████████████████████████████████  PIPELINE  ████████████████████████████████ -->

## 🔁 End-to-End Pipeline

```mermaid
flowchart TD
    RAW([📜 Shakespeare Corpus\n~1.1M characters]) --> TV

    subgraph PRE [🧹 Preprocessing]
        direction TB
        TV[TextVectorization\nadapt on full corpus]
        TV --> ENC[Integer encoding\n39 unique characters]
        ENC --> WIN[Sliding windows\nwindow_length = 100\nshift = 1]
        WIN --> DS[tf.data pipeline\nbatch + shuffle + prefetch]
    end

    subgraph MODEL [🧠 GRU Model]
        direction TB
        EM[Embedding layer\nvocab_size × 16]
        EM --> GRU[GRU 128 units\nreturn_sequences=False]
        GRU --> DN[Dense 39\nsoftmax activation]
        DN --> LO[sparse_categorical_crossentropy\nAdam optimizer]
    end

    subgraph TRAIN [📈 Training]
        direction TB
        T[10 epochs\n~3.8 hours on CPU]
        T --> AC[~60% train accuracy\n~55% val accuracy]
    end

    subgraph GEN [✍️ Text Generation]
        direction TB
        SEED[Seed string\ne.g. 'ROMEO: ']
        SEED --> TEMP[Temperature sampling\n0.01 · 1.0 · 100]
        TEMP --> OUT[Autoregressive output\none char at a time]
    end

    PRE --> MODEL
    MODEL --> TRAIN
    TRAIN --> GEN

    style PRE fill:#1a1a2a,stroke:#7C3AED,color:#ffffff
    style MODEL fill:#2a1a2a,stroke:#EC4899,color:#ffffff
    style TRAIN fill:#1a2a1a,stroke:#22C55E,color:#ffffff
    style GEN fill:#2a2a1a,stroke:#F59E0B,color:#ffffff
```

---

<!-- ████████████████████████████████  MODEL  ████████████████████████████████ -->

## 🤖 Model Architecture

```
Input: integer-encoded character sequence (window = 100)
    ↓
Embedding(vocab_size, 16)       — maps each char ID to a 16-dim dense vector
    ↓
GRU(128)                        — gated recurrent unit, captures sequential dependencies
    ↓
Dense(39, activation='softmax') — probability distribution over 39 unique characters
```

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=16),
    tf.keras.layers.GRU(128),
    tf.keras.layers.Dense(len(char_vocab), activation="softmax")
])

model.compile(
    loss     = "sparse_categorical_crossentropy",
    optimizer = "adam",
    metrics  = ["accuracy"]
)
```

---

<!-- ████████████████████████████████  TEMPERATURE  ████████████████████████████████ -->

## 🌡️ Temperature Sampling — Creativity vs Coherence

<div align="center">

| Temperature | Behaviour | Output Style |
|:---:|:---|:---|
| `0.01` | Near-deterministic — always picks the most likely character | Repetitive, highly coherent, but boring |
| `1.0` | Balanced — samples proportionally to learned probabilities | Natural-sounding Shakespearean text |
| `100` | Near-uniform — almost random character selection | Chaotic, creative, mostly incoherent |

</div>

```python
def next_char(text, temperature=1.0):
    encoded = preprocess([text])
    logits  = model.predict(encoded)[0][-1]
    scaled  = logits / temperature          # divide logits by temperature
    probs   = tf.nn.softmax(scaled)
    return tf.random.categorical([tf.math.log(probs)], num_samples=1)
```

> **Low temperature** = greedy, safe. **High temperature** = risky, creative. The right temperature is somewhere in between — usually 0.7–1.0 for convincing fake Shakespeare.

---

<!-- ████████████████████████████████  NOTEBOOKS  ████████████████████████████████ -->

## 📓 Notebooks

<div align="center">

| Notebook | Purpose |
|:---|:---|
| `03A_character_prediction_with_GRU.ipynb` | Reference implementation — based on Géron's *Hands-On ML* · full training pipeline |
| `Text_Generation.ipynb` | Student exploration — data analysis · anomaly findings · temperature experiments · annotated walkthrough |

</div>

### What We Found During Exploration

- **Corpus anomalies** — `$` and `3` appearing in Shakespeare text traced to **OCR errors / typos** in the digitised source
- **TextVectorization deep dive** — documented `adapt()`, `split`, and `standardize` parameters in detail
- **Training constraint** — model took 3+ hours on CPU, so used the pretrained weights for experimentation
- **Temperature experiments** — ran generation at `0.01`, `1.0`, and `100` to empirically show the creativity-coherence tradeoff

---

<!-- ████████████████████████████████  DATASET  ████████████████████████████████ -->

## 📊 Dataset

<div align="center">

| Property | Value |
|:---|:---|
| Source | Karpathy's `char-rnn` Shakespeare corpus |
| Size | ~1.1 million characters |
| Vocabulary | 39 unique characters (letters · punctuation · newline) |
| Encoding | Character-level integer mapping via `TextVectorization` |
| Window | 100-character sliding windows · shift = 1 |

</div>

---

<!-- ████████████████████████████████  RESULTS  ████████████████████████████████ -->

## 📈 Results

<div align="center">

| Metric | Value |
|:---:|:---:|
| Training Accuracy | ~60% |
| Validation Accuracy | ~55% |
| Training Time | ~3.8 hours (CPU) |
| Epochs | 10 |

</div>

> For character-level language modelling on a 39-token vocabulary, **~55% validation accuracy** is meaningful — the model has learned character co-occurrence patterns, word structure, and some syntactic style of Elizabethan English.

---

<!-- ████████████████████████████████  CONCEPTS  ████████████████████████████████ -->

## 📐 Key Mathematical Concepts

<div align="center">

| Concept | Application |
|:---|:---|
| **GRU (Gated Recurrent Unit)** | Update gate + reset gate — selectively retain/forget sequence history |
| **Character-Level LM** | Predict next character given previous 100 — vocabulary = 39 chars |
| **Embedding Layer** | Maps discrete char IDs → continuous 16-dim vectors |
| **Temperature Sampling** | Scales logits before softmax → controls output distribution sharpness |
| **Autoregressive Generation** | Feed predicted char back as input — generate one char at a time |
| **Sliding Window Dataset** | tf.data pipeline — 100-char windows, shift=1 for dense supervision |
| **Sparse Categorical Crossentropy** | Integer label loss — no one-hot encoding needed |

</div>

---

<!-- ████████████████████████████████  TECH  ████████████████████████████████ -->

## 🛠️ Tech Stack

<div align="center">

[![Tech](https://skillicons.dev/icons?i=tensorflow,python)](.)

| Library | Role |
|:---|:---|
| `tensorflow` / `keras` | Model definition · training · TextVectorization · tf.data |
| `tensorflow-cpu` | CPU-only build — no GPU required |
| `numpy` | Array operations |
| `matplotlib` | Training curves · accuracy / loss plots |
| Jupyter Notebook | Interactive development environment |

</div>

---

<!-- ████████████████████████████████  STRUCTURE  ████████████████████████████████ -->

## 🗂️ Repository Structure

```
ShakeGen/
├── 03A_character_prediction_with_GRU.ipynb   ← Reference notebook (Géron)
├── Text_Generation.ipynb                     ← Student exploration + annotations
├── images/                                   ← GitHub Classroom autograding screenshots
└── README.md
```

---

<!-- ████████████████████████████████  GETTING STARTED  ████████████████████████████████ -->

## 🚀 Getting Started

### 1️⃣ Install Dependencies

```bash
pip install tensorflow-cpu jupyter matplotlib numpy
```

### 2️⃣ Run the Notebooks

```bash
jupyter notebook
```

Open `Text_Generation.ipynb` to explore with the pretrained model, or `03A_character_prediction_with_GRU.ipynb` to run the full training pipeline.

> ⚠️ Full training takes ~3.8 hours on CPU. Use the pretrained model weights for experiments.

---

<!-- ████████████████████████████████  CONTEXT  ████████████████████████████████ -->

## 🎓 Course Context

<div align="center">

| | |
|:---|:---|
| 🏫 **Institution** | George Brown College |
| 📘 **Course** | AASD-4011 — Advanced Mathematics for Deep Learning |
| 📅 **Semester** | Winter 2023 |
| 📁 **Project Type** | Final paired project |
| 📚 **Reference** | Aurélien Géron — *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow* |

</div>

---

<!-- ████████████████████████████████  FOOTER  ████████████████████████████████ -->

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=1,8,15&height=120&section=footer" width="100%"/>

**Anisha Singla** · George Brown College · AASD-4011 Advanced Math for Deep Learning

[![TF](https://img.shields.io/badge/TensorFlow-GRU_RNN-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Shakespeare](https://img.shields.io/badge/Dataset-Shakespeare_1.1M-7C3AED?style=flat-square)](.)

> *"To GRU or not to GRU — that is the character prediction."*

⭐ Star this repo if it was useful!

</div>
