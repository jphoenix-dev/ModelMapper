# Glossary

Plain-English definitions, added as terms come up.

## How a model is built

- **Neuron:** one number the model computes at one spot for each word: a weighted sum of earlier numbers, then put through a *bend*. "Firing" means the number is big.
- **Bend (nonlinearity / activation function):** a simple rule applied to each neuron's sum that stops the output being a straight-line function of the input. ReLU is the simplest bend: "if negative, output zero". Without bends, any stack of layers collapses into one big multiplication and can only learn straight-line relationships. Because the bend is applied to each neuron separately, individual neurons are a natural unit to study.
- **Activation:** the value a neuron takes on a particular input. "Strong activations" are the inputs where it lights up most.
- **Weights (parameters):** the model's fixed, learned numbers, set once by training. Think of them as the wiring of a circuit board. Activations are the currents that flow through that wiring when you feed in a particular text. We only ever read or overwrite activations, never weights.
- **Fires hard:** large *relative to that neuron's own peak* on ordinary web text, since every neuron has its own scale. Our working bar: a Golden Gate candidate must reach ≥ 20% of its peak on Golden Gate texts, and test sentences written from an explanation must reach ≥ 30%.
- **Clamp:** force a neuron to a fixed value on every word during a run (e.g. 8× its peak). It's the mechanism behind steering (high value) and ablation (zero). Temporary: nothing changes once the run ends.
- **Fine-tuning:** further training that changes the weights permanently. Not something we do to Qwen.
- **Layer / MLP:** a model is a stack of layers. Each layer has an *attention* part, which lets each word look back at earlier words, and an *MLP* part: a bank of thousands of neurons applied to each word. "Neurons" here means MLP neurons, and "MLP width" means how many there are.
- **Token:** the unit of text the model reads, usually a word or a piece of a word ("Golden", " Gate", "iy"). Roughly 3/4 of a word on average.
- **SwiGLU:** the bend used in modern LLMs. Each neuron multiplies a soft on/off *gate* by a *value* that can be positive or negative, so the neuron's output can go either way. Formula: output = SiLU(gate) × value.
- **SiLU:** the gate's bend. Like ReLU but smooth: big positive in → passes through; negative in → a small negative number (never below about −0.28).
- **Sign regimes / double negative:** because SiLU(gate) can be slightly negative, a negative gate times a negative value gives a *positive* output. So a neuron can fire positive for two different reasons. We tag each firing by (gate sign, value sign) so we don't confuse them.
- **Poles:** the positive and negative ends of a neuron's output, treated as separate jobs.
- **Weighted sum (dot product):** multiply each number in a list by its own weight and add the products, giving one number. It measures how much the list "points the same way" as the weights: large positive = aligned, ~0 = unrelated, negative = opposite. Gates, values and the final next-word scores are all weighted sums.
- **Softmax:** turns the final scores for every vocabulary word into probabilities that add up to 100%; a modest lead in score becomes a big lead in probability.
- **Dense model:** every neuron reads the whole belt (all 4,096 numbers in Qwen3-8B) for every word, even if it ends up near zero. About one multiply-add per weight per word: ~5 billion per word for Qwen3-8B.
- **Mixture-of-experts (MoE):** each layer's neurons are split into many "experts", and a small router sends each word to only a few of them; the rest don't run. Qwen3.6-35B-A3B has 35B weights but only ~3B Active per word. The essay calls experts "anti-superposition": the router would have to recognise every unrelated thing packed into an expert.
- **Quantisation:** storing weights with fewer bits (e.g. 4 instead of 16), rounding each to a coarser grid. It shrinks the model and speeds it up; errors largely cancel inside the weighted sums, but weak activations get relatively noisier. We use unquantised bf16 so neuron readings are precise.
- **Residual stream:** the conveyor belt running through the model. Each layer adds its contribution rather than replacing what's there, so information can bypass a layer.
- **Down projection:** the weights that turn a layer's neuron outputs back into a contribution to the residual stream. Each neuron has its own column: when the neuron fires, that column (scaled) is what it adds to the conveyor belt.

## The debate

- **Polysemantic / monosemantic:** a neuron that responds to seemingly unrelated things (polysemantic) versus one thing (monosemantic).
- **Superposition:** the theory that a model crams unrelated concepts into shared neurons because those concepts rarely occur together. Analogy: two radio stations sharing one frequency because they never broadcast at the same time.
- **Recursive generalisation:** a neuron's job drifting in locally sensible steps until its end points look unrelated.
- **Summary motif:** neurons often fire on punctuation (commas, full stops), where the model stores a running summary of the context. Read the surrounding text, not just the token.
- **Selection bias:** getting a skewed answer because of *how you chose* what to look at. If we go hunting for a clean Golden Gate neuron and find one, that tells us little about whether a *random* neuron is clean.

## The tools

- **Direction / rotation / basis:** a layer's output is a list of N numbers, which you can treat as a point in N-dimensional space where each neuron is one axis (the "neuron basis"). A direction is any mix of neurons; a rotation is a different choice of axes.
- **SAE (sparse autoencoder):** a second network trained to re-describe a layer's output using a few entries from a much larger dictionary of directions. It is meant to unpack superposed concepts. Golden Gate Claude (Anthropic, 2024) was made by turning up one SAE dictionary entry.
- **Loss:** the model's error score at predicting the next word. Training pushes it down.
- **Ablate / steer:** switch a unit off (ablate) or turn it up (steer) and see what changes. These are the causal tests.
- **Hook:** a small tap we attach inside the model to read (or overwrite) numbers partway through a forward pass, without changing the model's code.
- **Forward pass:** running text through the model once, start to finish, to get its prediction.
- **Top-k examples:** the k text snippets (from a big pile of text) where a neuron fired most strongly. The standard first look at a neuron.
- **Contrastive search:** finding candidate neurons by comparing activations on texts *with* a concept against matched texts *without* it, then ranking neurons by the difference.
- **Near miss:** a test example deliberately close to, but outside, an explanation (e.g. "the Bay Bridge" for a Golden Gate neuron). A good explanation should predict the neuron stays quiet on it.
- **Blind maximum:** one agent writes an explanation; a second sees only the explanation and writes the text it thinks will fire the neuron hardest. If that beats the best real example, the explanation captured the neuron's centre.
- **Intruder detection:** show a judge several of a neuron's top examples plus one random "intruder" and ask which doesn't belong. No explanation is needed, so it can't be fooled by a good story. If the judge spots the intruder easily, the examples share something real.
- **Simulation score:** an explanation is used to predict activations on new text, and the prediction is correlated with the truth. The essay distrusts these as a primary metric.
- **Control / baseline:** a comparison case run through exactly the same pipeline (e.g. a random neuron or a random mix of neurons), so we can tell whether a result is about our target or about our method.
- **Checkpoint:** a saved snapshot of a model partway through training.
- **TinyStories:** a dataset of simple children's stories that tiny models can learn to write coherently.
- **Jev:** TypeSafe's "System One" model. It doesn't write text; it answers fixed-choice questions (yes/no, pick one, score) with calibrated probabilities. It's cheap (~$0.04 per million input tokens) and already used in Neuronpedia-style neuron scoring. We use it as a fast, blind judge.

## Measurements and methods (added 16 Oct, overnight)

- **Recall prompt:** a question whose answer is the concept ("What is Earth's only natural satellite called?" →
  Moon). If the model gives the answer with decent probability, it "recalls" the concept, and we can look inside to
  see which neurons made that happen.
- **Completion vs question format:** completion = an unfinished sentence ("The tallest mountain on Earth is
  Mount"); question = a chat question answered at the start of the reply. Formats change the numbers, so we report
  which one we used.
- **Answer discovery:** let the model answer the recall question itself first, and score *its* wording ("Rainbow",
  "Natural selection") rather than ours ("rainbows"). Made-up concepts must still come out "not recalled".
- **Made-up controls:** realistic fake concepts (a peak, a linguist, a bridge that don't exist). Any method that
  "finds" them is broken. They caught one bug tonight.
- **Attribution (activation × gradient):** a fast estimate of how much each neuron contributed to the answer. The
  neuron's value, times how much the answer's probability would change if that value moved. It ranks candidates; it
  is an estimate, not a measurement.
- **Measured single-ablation effect:** the slow, direct version. Switch one neuron off, re-run, see how much the
  answer drops. Used to check and re-rank attribution.
- **k50:** fewest neurons you must switch off to halve the probability of the answer. **Saturates:** when the model is
  99.8% sure, halving it takes a huge push, so k50 looks big even if a few neurons matter a lot.
- **Log-odds:** log(p / (1 − p)), a scale where every step means the same thing at any confidence level. A drop of 3 =
  the odds fall ~20×. Our main "how concentrated is recall" measure now.
- **Flip:** the answer actually changes (probability falls below 50%) on a prompt the model was getting right.
- **Self-repair (re-routing):** when you switch neurons off, others that weren't doing much start carrying the answer.
  Happens for every concept we tested.
- **Clamp vs add steering:** clamp forces a neuron to a fixed value; add nudges it up on top of whatever it was
  doing. Add is gentler and slightly better for fluency.
- **Reply-only vs prompt steering:** steer only while the model writes its answer, or also while it reads the
  question. Steering the question too is stronger; we report the harder reply-only version.
- **Nudgeability (joint):** share of neutral prompts where the steered reply is about the concept **and** still
  fluent, judged on the same reply. **Lenient** judge = "mentions or seems preoccupied with X"; **strict** = "X is
  the main topic". Strict roughly halves scores.
- **Neuron-picking rules (R1–R5):** R1 top attributed in late layers (the original rule); R2 any layer; R3 measured
  single-ablation; R4 neurons that directly push the answer's word ("says the answer"); R5 neurons whose label names
  the concept. Different concepts prefer different rules.
- **Direct vocabulary readout:** follow a neuron's output weights straight to the final word scores to see which words
  it pushes, without running any text.
- **Winner's curse:** if you try 15 settings and report the best on the same prompts, you flatter the number. Fix:
  re-test the chosen setting on fresh **held-out** prompts.
- **Prompt-set noise:** how much a number changes if you re-run with different (independently written) prompts.
  Measured tonight: ± ~0.1 on a single concept's steering score.
- **Bootstrap 95% CI:** resample your own data many times to see how much a number would wobble. A range, not a
  single value.
- **AUC:** how well a score separates two groups (1.0 = perfectly, 0.5 = coin flip). Used for "does the grader tell
  real examples from random text".
- **Calibration / spliced dossiers:** to set the "clean" bar, we built fake two-job neurons (examples from two
  unrelated neurons mixed together) and chose the bar that lets ≤ 10% of them through.
- **Test-retest:** run the identical labelling twice; how often does the verdict change? (~20% per neuron; the
  population rate is stable.)
- **Pre-registration:** writing down predictions and the exact method *before* running, so we can't move the
  goalposts. Tonight's extra runs are exploratory; CR-3 would make them confirmatory.
- **Greedy decoding:** the model always picks its single most likely next word, so the same input gives the same
  output (reproducible).
