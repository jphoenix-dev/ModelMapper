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
