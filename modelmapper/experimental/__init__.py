"""EXPERIMENTAL: real-text checks of labels (the corpus exam and graded simulation, LABELLING.md "Experimental").

This is the research code that ran on Qwen3-8B (21 Oct 2026: corpus exam v2 on 12,244 labelled neurons; graded
simulation on every labelled neuron's gate and side labels), packaged with its paths and model made settings. It has not been re-run or calibrated in this
package beyond offline tests, and its interfaces may change.

    realtext.py      corpora, the corpus readings pass (each neuron's strongest real-text moments), labels to check
    corpus_exam.py   the corpus exam: positives against same-token, same-topic and random decoys
    simulate.py      graded simulation: the judge rates how strongly a neuron fires at stratified positions
"""
