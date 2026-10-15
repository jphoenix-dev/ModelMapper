# Experiment GG summary: Qwen3-8B

Scan: 10,005,329 tokens.

## By unit type

| type | n | pos clean | neg clean | pos verdicts (ONE/TWO/MIX/NONE) | neg verdicts | intruder P pos | intruder P neg |
|---|---|---|---|---|---|---|---|
| candidate | 24 | 9 | 6 | 16/1/4/3 | 14/2/4/4 | 0.40 | 0.39 |
| exploratory | 8 | 5 | 5 | 7/0/0/1 | 6/0/0/2 | 0.45 | 0.46 |
| random_neuron | 20 | 10 | 3 | 16/2/0/2 | 9/4/3/4 | 0.39 | 0.33 |
| planted_mixture | 10 | 0 | 1 | 5/0/4/1 | 2/1/5/2 | 0.29 | 0.32 |

## Every unit

### L10_N838 (candidate_pos_early, dossier U132)
- intruder P pos/neg: 0.419/0.324 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 17/20, random fit 0/20, clean=True: The firings cluster around relational complements, especially adjective + “to” constructions that say something is relevant, characteristic, suitable, or emotionally important to a person or context (for example, “dear to,” “common to,” and “endemic to”).
- **neg** [MIXTURE] covered 0/20, random fit 0/20, clean=False: There is no clear shared job: the examples include legal uses of “direct/indirect,” several unrelated uses of “lovingly,” and a broad assortment of narrative, descriptive, and inflected word forms such as “brave,” “solemnized,” “playful,” and “lasting.”

### L11_N11527 (candidate_neg_early, dossier U323)
- intruder P pos/neg: 0.447/0.443 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 20/20, random fit 1/20, clean=True: Fires on U.S. government and public-policy reporting: legislation, federal agencies and funding, and government actions or agreements in diplomacy. The examples include both domestic policy and foreign affairs involving the U.S.
- **neg** [ONE] covered 11/20, random fit 0/20, clean=False: Favors international-affairs and postwar geopolitical material, including UN institutions, wars and military history, and the European trade and industrial arrangements that followed World War II.

### L12_N2637 (candidate_neg_middle, dossier U338)
- intruder P pos/neg: 0.757/0.531 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 19/20, random fit 0/20, clean=True: The unit marks explicit conditions or requirements that must be satisfied, overwhelmingly casino-bonus wagering and playthrough thresholds before winnings can be withdrawn or bonuses qualify. The Cub Scout rank requirements are a non-gambling instance of the same general idea.
- **neg** [ONE] covered 18/20, random fit 0/20, clean=True: The unit marks expressions of necessity or obligation—such as required, necessary, need, essential, and have to—often describing a prerequisite for a policy, task, or outcome.

### L12_N9831 (candidate_neg_middle, dossier U617)
- intruder P pos/neg: 0.446/0.442 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 11/20, random fit 0/20, clean=False: The clearest pattern is health and biomedical language, especially disease, symptoms, treatments, and clinical research; several firings land inside technical medical terms. Some firings are unrelated names or general-interest topics.
- **neg** [ONE] covered 17/20, random fit 1/20, clean=True: This pole strongly favors US geographic names—especially counties, towns, and other local places—often in regional news, listings, or descriptions. Two firings are technical terms rather than place names.

### L13_N2962 (candidate_pos_middle, dossier U674)
- intruder P pos/neg: 0.459/0.321 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [MIXTURE] covered 11/20, random fit 0/20, clean=False: The main pattern is saline, ocean, and coastal-water contexts: saltwater fishing or intrusion, salinity, sea-level rise, and storm surges. There are also unrelated firings on Boston Tea Party references and on the token in “Capricorn.”
- **neg** [ONE] covered 14/20, random fit 0/20, clean=False: This pole mostly fires on geographic location information, from latitude/longitude or ship-position data to named US regions, states, and place names in weather reports or addresses. It does not fit the fictional names or software-path example.

### L13_N7489 (candidate_pos_middle, dossier U127)
- intruder P pos/neg: 0.572/0.304 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 20/20, random fit 0/20, clean=True: Fires on the distinguishing name element in formal names of universities, often immediately before “University” (e.g. Louisiana State, Wesleyan, Concordia, and Dalhousie).
- **neg** [NONE] covered 0/20, random fit 0/20, clean=False: The negative examples do not show a stable shared pattern: the marked tokens occur in unrelated contexts ranging from recipes and clothing to animal breeds, product names, and technical specifications.

### L14_N7392 (candidate_pos_middle, dossier U765)
- intruder P pos/neg: 0.273/0.4 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [NONE] covered 0/20, random fit 0/20, clean=False: The marked tokens occur in a wide range of unrelated contexts, from finance and software to travel, recipes, and education. I can’t identify a shared semantic or contextual feature that explains the firings.
- **neg** [NONE] covered 0/20, random fit 0/20, clean=False: These firings also span unrelated material, including medical writing, entertainment, recipes, software instructions, sports, and geography. The examples don’t support a reliable shared feature for this pole.

### L15_N3285 (candidate_neg_middle, dossier U325)
- intruder P pos/neg: 0.322/0.3 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [MIXTURE] covered 4/20, random fit 0/20, clean=False: Two unrelated clusters recur: Creative Commons license boilerplate (Noncommercial/ShareAlike) and promotional descriptions or plans for apps and Apple devices, especially iPod touch. The other examples do not clearly fit either pattern.
- **neg** [NONE] covered 2/20, random fit 0/20, clean=False: The firings range across ordinary words, place and personal names, technical vocabulary, and partial Polish word fragments. There is no reliable shared pattern in the examples.

### L16_N1944 (candidate_neg_middle, dossier U384)
- intruder P pos/neg: 0.57/0.46 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 20/20, random fit 0/20, clean=True: Fires on people’s surnames or final name tokens in biographies, news, quotes, and other references to individuals. The examples span many fields, but the shared pattern is a person-name mention.
- **neg** [ONE] covered 13/20, random fit 2/20, clean=False: Often fires on names of media outlets, publications, shows, or institutions rather than on a person’s name. Several examples are personal names or technical text instead, so that pattern does not cover the whole pole.

### L16_N9424 (candidate_pos_middle, dossier U703)
- intruder P pos/neg: 0.37/0.431 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [NONE] covered 0/20, random fit 2/20, clean=False: The marked words occur in unrelated subject areas and do not show a stable semantic, stylistic, or grammatical pattern.
- **neg** [NONE] covered 0/20, random fit 0/20, clean=False: These firings likewise span unrelated topics, from cars and flowers to medical research and email marketing, without a shared pattern evident in the marked words or their contexts.

### L29_N1596 (candidate_pos_late, dossier U444)
- intruder P pos/neg: 0.315/0.273 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 15/20, random fit 1/20, clean=False: The clearest pattern is fragments of proper names and named places, venues, works, or game locations, often in directory or encyclopedic text. A few marked tokens are ordinary grammatical words or noisy Chinese text rather than name pieces.
- **neg** [ONE] covered 8/20, random fit 0/20, clean=False: Many firings land on personal names, especially given names or name components in biographies, interviews, news, and event coverage. Other marked tokens are punctuation, code, ordinary words, or numeric label text, which do not fit that pattern.

### L29_N3554 (candidate_neg_late, dossier U467)
- intruder P pos/neg: 0.287/0.255 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 7/20, random fit 0/20, clean=False: The clearest pattern is scalar degree or extent: intensifiers such as Chinese 很, and phrases like “too dirty,” “too shallow,” “too wide/narrow,” and “just as hot.” Several other firings land on ordinary words in contexts that do not show this pattern.
- **neg** [TWO_RELATED] covered 10/20, random fit 4/20, clean=False: These firings mostly land on orthographic or formatting-heavy material rather than a shared meaning: digits and punctuation in dates, citations, measurements, and sequences, alongside Vietnamese text and pieces of names or technical terms. The marked “the” in example 12 does not fit that pattern.

### L29_N5086 (candidate_neg_late, dossier U199)
- intruder P pos/neg: 0.394/0.172 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 19/20, random fit 1/20, clean=True: Fires mainly within proper names—especially brands, retailers, people, places, and product or film titles. It can also fire on a conjunction or punctuation embedded in a name or title.
- **neg** [ONE] covered 11/20, random fit 0/20, clean=False: Usually fires on “work” and its forms, or nearby wording about employment, job duties, or putting effort into something. Some examples hit a function word immediately before “work,” rather than the word itself.

### L29_N6105 (candidate_pos_late, dossier U489)
- intruder P pos/neg: 0.305/0.352 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 8/20, random fit 0/20, clean=False: Primarily environmental sustainability and conservation material: recycling and e-waste, climate action, clean water, and sustainable products or infrastructure. Example 4 also fits as reporting on wildfire-related erosion and flooding hazards.
- **neg** [TWO_RELATED] covered 0/20, random fit 0/20, clean=False: Most examples concern home loans and mortgages, including loan types, rates, eligibility, property requirements, and fees. Examples 8 and 13 instead concern municipal land-use planning, zoning, and building review.

### L29_N8837 (candidate_pos_late, dossier U194)
- intruder P pos/neg: 0.415/0.482 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [MIXTURE] covered 5/20, random fit 0/20, clean=False: The firings split into unrelated clusters: forms containing “fall” (including place names and names such as Fallbrook, Fallujah, and Fallon), Boost C++ namespace syntax (`boost::`), and technical or descriptive words ending in “-al” such as Operational and Imperial. The examples do not suggest one shared function connecting these patterns.
- **neg** [ONE] covered 17/20, random fit 0/20, clean=True: These examples all come from a technical tutorial on implementing a floating-point adder, especially IEEE-style special-case handling and related circuit logic. The later example on aligning operands is part of the same implementation discussion.

### L31_N7303 (candidate_pos_late, dossier U370)
- intruder P pos/neg: 0.233/0.336 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [TWO_RELATED] covered 16/20, random fit 0/20, clean=False: The main pattern is technical, process-oriented writing: software/server setup instructions and computational-science discussion of methods, parameters, and results. The SQL tutorial fits the same practical technical-documentation style; the car and travel passages do not.
- **neg** [MIXTURE] covered 0/20, random fit 0/20, clean=False: This pole is heterogeneous: several firings occur in formal research or scientific summaries, while others occur in travel, lifestyle, food, local-news, and general informational writing. I don't see a reliable shared content pattern beyond these disparate kinds of expository web text.

### L33_N12018 (candidate_neg_late, dossier U649)
- intruder P pos/neg: 0.432/0.383 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 10/20, random fit 1/20, clean=False: Mostly fires on capitalized word starts or labels—such as product names, headings, and proper nouns—rather than on a shared subject matter. A few firings land on lowercase fragments or a word ending, so the capitalization pattern is approximate.
- **neg** [ONE] covered 4/20, random fit 4/20, clean=False: Mostly fires on lowercase word pieces or spellings in running text, including common nouns and technical terms; there is no evident shared topic. Some capitalized ingredient or name tokens are exceptions to this lowercase tendency.

### L34_N119 (candidate_neg_late, dossier U487)
- intruder P pos/neg: 0.35/0.337 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 16/20, random fit 0/20, clean=True: The pole is associated with door-related wording: literal doors and doorways, compounds such as door-to-door, indoors/outdoors and doorstep, and figurative “opening doors” for access or opportunity. Some firings fall on nearby words rather than on “door” itself.
- **neg** [ONE] covered 15/20, random fit 0/20, clean=False: This pole tracks the written gate/gated/tailgate family across different contexts: scandal labels, technical or mechanical terms, named places and products, and the idiom “out of the gate.” The shared pattern is lexical form, not one consistent meaning of gate.

### L6_N4064 (candidate_pos_early, dossier U214)
- intruder P pos/neg: 0.357/0.317 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 6/20, random fit 0/20, clean=False: Most firings fall on proper names, organizations, and place names, especially San Francisco and Golden Gate references; many are marked within the name itself. A few firings land on ordinary words or technical text instead.
- **neg** [MIXTURE] covered 0/20, random fit 0/20, clean=False: This pole is a mixture of unrelated patterns: occurrences in Talbot/Talbots names and places (1, 3, 4, 6, 19), TGC/TMG abbreviations (5, 10, 11, 13, 16, 17, 20), technical compounds such as bandpass and trust chain (7, 9, 14, 18), and coin-flip comparisons (8, 15). Example 12 is a separate finance phrase, “capital structure.”

### L7_N10008 (candidate_neg_early, dossier U242)
- intruder P pos/neg: 0.359/0.581 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 10/20, random fit 2/20, clean=False: The pole most strongly tracks commas, especially commas separating names, credentials, dates, citations, or items in a list. The examples span many subjects, so the punctuation pattern is more consistent than any topic.
- **neg** [ONE] covered 20/20, random fit 3/20, clean=True: The pole tracks periods/full stops, often at the end of a sentence or a scraped snippet. The surrounding material ranges from spam and garbled text to coherent articles, so the punctuation boundary is the clearer shared pattern.

### L7_N10114 (candidate_neg_early, dossier U854)
- intruder P pos/neg: 0.337/0.226 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [MIXTURE] covered 6/20, random fit 0/20, clean=False: There is a cluster of marked tokens in idioms or set phrases (e.g. “larger than life,” “jump the shark,” “over the hill”), alongside separate clusters on the unusual word “hugelkultur” and the casino slogan “Hit It Rich.” The other firings look like unrelated fragments rather than one shared feature.
- **neg** [MIXTURE] covered 1/20, random fit 1/20, clean=False: The firings mix proper-name or place-name fragments with webpage-interface remnants such as login text and separators. “Global Green” recurs in the school article, but it does not explain the other examples.

### L7_N1581 (candidate_pos_early, dossier U328)
- intruder P pos/neg: 0.351/0.593 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [NONE] covered 3/20, random fit 0/20, clean=False: There is no stable semantic or spelling pattern across these firings: they fall on assorted fragments in personal names, Roman numerals and product identifiers, and ingredient or library strings.
- **neg** [ONE] covered 16/20, random fit 0/20, clean=True: This pole most consistently fires within words beginning with “poly-,” including Polycarp, polychrome, polycarbonate, polycystic, and polyamorous. The shared pattern is spelling, not a single subject matter.

### L7_N4091 (candidate_pos_early, dossier U350)
- intruder P pos/neg: 0.283/0.386 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 18/20, random fit 2/20, clean=True: The marked pieces mostly occur inside identifying strings: personal or organization names, acronyms, and structured references such as phone numbers and an ISBN. The firing appears tied more to these name/label contexts than to a shared subject matter.
- **neg** [ONE] covered 7/20, random fit 0/20, clean=False: The strongest negative pattern is the final “ru” subword in “Nehru,” recurring in contexts about Nehru Stadium, the university, or Jawaharlal Nehru. The remaining examples do not show a clear connection to that pattern.

### L8_N4590 (candidate_neg_early, dossier U189)
- intruder P pos/neg: 0.493/0.6 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 20/20, random fit 1/20, clean=True: Fires around framing of claims and opinions circulating in public discourse—especially conventional wisdom, punditry, rhetoric, and advice that people hear or repeat. The examples span topics, but share this focus on received or publicly expressed views.
- **neg** [ONE] covered 19/20, random fit 0/20, clean=True: Strongly associated with quantified prevalence statements, particularly percentages or fractions followed by “of,” as in statistics about populations, behaviors, or outcomes.

### L17_N8016 (exploratory_unnamed_neg, dossier U877)
- intruder P pos/neg: 0.217/0.254 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [NONE] covered 4/20, random fit 0/20, clean=False: The marked positions range from fragments of exam IDs and punctuation to ordinary words in unrelated prose, recipes, technical text, and FAQs. The repeated certification-spam snippets may explain a few firings, but do not account for the rest; no stable shared behavior is evident.
- **neg** [NONE] covered 2/20, random fit 1/20, clean=False: The negative firings include many different grammatical words and fragments, appearing across unrelated subjects and text styles. There is no consistent semantic or syntactic pattern that explains them as a group.

### L21_N6412 (exploratory_unnamed_neg, dossier U263)
- intruder P pos/neg: 0.365/0.509 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 17/20, random fit 0/20, clean=True: This pole tracks color specifications and color/material references, especially cool or neutral shades such as teal, turquoise, gray, black, blue, and silver; several firings are inside RGB or hex color codes. Most examples are explicit color descriptions, though a few appear to rely on an indirect visual or material association.
- **neg** [ONE] covered 16/20, random fit 0/20, clean=True: This pole is associated with warm, vivid color references, especially red, scarlet, and pink, including color-coded foods, flowers, hair, and clothing. Some firings are on names or terms that may evoke those colors, but are not explicitly color descriptions in the excerpt.

### L25_N6245 (exploratory_unnamed_pos, dossier U259)
- intruder P pos/neg: 0.737/0.201 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 20/20, random fit 0/20, clean=True: This pole responds to citrus and citrus-color words—especially “orange” and “lemon,” plus related forms like “tangerine”—whether they refer to a color, fruit, name, or title.
- **neg** [NONE] covered 6/20, random fit 0/20, clean=False: I can’t identify a reliable shared meaning: the firings include fragments inside proper names, ordinary words, punctuation, a date, and “palm” in unrelated contexts.

### L25_N8657 (exploratory_unnamed_neg, dossier U320)
- intruder P pos/neg: 0.343/0.367 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 12/20, random fit 0/20, clean=False: The pole is associated mainly with washing, bathing, and ablution—both ritual washing and ordinary cleaning such as washing dishes or hands. It does not appear to cover water-related scenes in general.
- **neg** [ONE] covered 18/20, random fit 0/20, clean=True: This pole tracks descriptions of visual appearance or perception: things seen from a distance, becoming visible, or compared to shapes such as a speck, an ant army, or a snake. The comparisons are common but not required; several examples describe visible lines, forms, or scenery directly.

### L26_N1311 (exploratory_unnamed_pos, dossier U881)
- intruder P pos/neg: 0.72/0.6 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 20/20, random fit 0/20, clean=True: Fires on the word “orange” and close forms or names containing “orang,” whether it is a color/food term, part of a proper name, or the title of a show. This also includes “orangery” and “orangutan.”
- **neg** [ONE] covered 19/20, random fit 0/20, clean=True: The marked fragments mostly occur inside proper names, especially place names and venues, often in less-common spellings; the surrounding topics vary widely, so this is a tentative orthographic/name pattern rather than a subject-matter theme.

### L3_N4061 (exploratory_unnamed_neg, dossier U754)
- intruder P pos/neg: None/0.719 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 39
- **pos** [ONE] covered 1/20, random fit 0/20, clean=True: The firing is on “Orono,” a place name in Maine; the surrounding text identifies it as the University of Maine’s location. With only one example, this is a tentative place-name hypothesis.
- **neg** [ONE] covered 20/20, random fit 0/20, clean=True: The pole fires on the word “orange” and closely related forms such as “orangery” and “Orangerie,” across uses as a color, fruit/flavor, place name, title, or name. The shared pattern appears lexical rather than tied to one meaning.

### L6_N1860 (exploratory_unnamed_pos, dossier U125)
- intruder P pos/neg: 0.344/0.611 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 12/20, random fit 0/20, clean=False: The firing tends to fall on the noun or material immediately after a color adjective, especially in product, fashion, and décor descriptions (e.g. “pink patent,” “purple plastic,” “gray wall”). Some examples are less commercial, but still use the same color-modifier-before-noun construction.
- **neg** [ONE] covered 19/20, random fit 0/20, clean=True: The firing is on color words used as parts of color combinations: coordinated lists or palettes, and compound or blended color descriptions (such as “gray-white,” “greenish black,” and “orangish red”).

### L8_N1847 (exploratory_unnamed_pos, dossier U658)
- intruder P pos/neg: 0.424/0.437 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 20/20, random fit 0/20, clean=True: Fires on color names and color-coded combinations, especially when colors are being selected, described, or used as a palette; this includes team colors and color codes in crafting examples.
- **neg** [ONE] covered 15/20, random fit 0/20, clean=False: Fires when color-like strings occur without describing a color choice or palette—often in proper names, idioms, or other lexicalized expressions (such as “White House,” “Great White,” and “cyanide”).

### mix_L10_N838_N8989 (planted_mixture, dossier U195)
- intruder P pos/neg: 0.262/0.333 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 13/20, random fit 0/20, clean=False: The clearest pattern is an adjective or adjective-like expression relating a quality, significance, or fit to a person, group, or thing—often in constructions such as “dear to,” “common to,” “endemic to,” or “characteristic of.” This also covers the contrasting relation in “anathema to.”
- **neg** [NONE] covered 0/20, random fit 0/20, clean=False: I don’t see a stable pattern across these firings: they range from function words and punctuation to unrelated content words. The repeated “stop at nothing” phrase in 2 and 10 is too small a cluster to account for the rest.

### mix_L13_N2962_N10947 (planted_mixture, dossier U833)
- intruder P pos/neg: 0.362/0.301 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [MIXTURE] covered 3/20, random fit 0/20, clean=False: This pole is mixed: it fires on formalized or standardized frameworks—clinical guidelines, design patterns, conformance, shared templates and software practices—and separately on health-related language such as hay fever, allergies and fussy eating. Those are distinguishable patterns, not one clear job.
- **neg** [ONE] covered 16/20, random fit 0/20, clean=True: This pole consistently marks bespoke, independently managed or fragmented approaches: custom or proprietary systems, ad hoc methods, disparate tools, and people making or tracking something on their own. These are often contrasted with coordinated, shared or integrated alternatives.

### mix_L13_N7489_N11636 (planted_mixture, dossier U303)
- intruder P pos/neg: 0.218/0.341 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [NONE] covered 0/20, random fit 1/20, clean=False: The marked positions range across punctuation, line breaks, word fragments, and ordinary words in unrelated contexts. There is no recurring pattern specific enough to explain the firings.
- **neg** [MIXTURE] covered 9/20, random fit 0/20, clean=False: A substantial cluster fires on count or container words in ingredient and quantity lists (such as “balls,” “pairs,” “bunch,” “leaf,” and “stalks”). The remaining firings are scattered names and technical or descriptive terms without a clear shared job.

### mix_L14_N7392_N11791 (planted_mixture, dossier U529)
- intruder P pos/neg: 0.215/0.231 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 15/20, random fit 12/20, clean=False: The examples mostly come from descriptive, reporting, or explanatory web prose, including property descriptions, news, reviews, and technical information. This contrasts with the strongly stepwise recipe and craft instructions prominent at the negative pole.
- **neg** [TWO_RELATED] covered 12/20, random fit 0/20, clean=False: Firing is prominent in segmented instructions, especially numbered recipe and craft steps, and also appears in structured exercise and ranking/list material. These share a formatted, sequential presentation, rather than continuous descriptive prose.

### mix_L16_N9424_N6534 (planted_mixture, dossier U106)
- intruder P pos/neg: 0.303/0.434 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 11/20, random fit 0/20, clean=False: This pole strongly tracks California Proposition 65 product warnings, especially boilerplate naming a chemical, California, cancer or reproductive harm, and the P65Warnings website. Several firings land directly on “65,” “Warnings,” or wording within that warning.
- **neg** [MIXTURE] covered 0/20, random fit 0/20, clean=False: The negative firings are scattered across unrelated topic-specific words and phrases: flower remedies, construction, geography and cars, medical/scientific topics, marketing, and other subjects. No consistent shared function is apparent.

### mix_L29_N1596_N18 (planted_mixture, dossier U718)
- intruder P pos/neg: 0.22/0.214 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 8/20, random fit 0/20, clean=False: The pole most consistently fires in solution-framing phrases, especially “the answer” and “the key,” sometimes with a verb such as “hold” beforehand. Several other firings appear to be incidental token or name fragments rather than part of this pattern.
- **neg** [NONE] covered 3/20, random fit 2/20, clean=False: There is no stable pattern I can identify: the firings range from name fragments and forum boilerplate to code punctuation, numbers, and ordinary prose. These examples do not support a single coherent hypothesis for this pole.

### mix_L31_N7303_N10244 (planted_mixture, dossier U570)
- intruder P pos/neg: 0.222/0.327 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [MIXTURE] covered 0/20, random fit 0/20, clean=False: The examples show several unrelated contexts: flood aftermath and damage to homes, roads, bridges, and crops; a software installation tutorial; and a molecular-simulation mailing list. The firings in the latter two often land on formatting or incidental tokens, so there is no clear single semantic feature tying the clusters together.
- **neg** [ONE] covered 9/20, random fit 0/20, clean=False: A substantial cluster concerns engineered structures and earthworks: dams, embankments, slopes, walls, mining earth movement, farm conservation structures, and construction or landscaping. Other examples are about unrelated topics such as ocean modeling, health, religion, and travel.

### mix_L6_N4064_N1528 (planted_mixture, dossier U381)
- intruder P pos/neg: 0.505/0.374 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 10/20, random fit 0/20, clean=False: The pole strongly tracks the spelling “Akash” in proper names and especially in “Akashic Records” material, including Akash, Saakashvili, and Prakash. The shared evidence is primarily orthographic rather than a single topic.
- **neg** [MIXTURE] covered 0/20, random fit 0/20, clean=False: There are several unrelated clusters: Talbot/Tarbotton names and locations (1, 4, 6, 7), repeated “acanthus” references in gardening and decorative design (9–11, 14, 16), and “evolution” in a book description (3, 13). These do not form one clear shared job.

### mix_L7_N1581_N230 (planted_mixture, dossier U858)
- intruder P pos/neg: 0.243/0.31 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [MIXTURE] covered 10/20, random fit 0/20, clean=False: The firings split between ellipsis punctuation in narrative text and fragments of proper names or specialized labels (for example, Gaultier, Barr, MPS VI, and polysorbate). These do not suggest one common function.
- **neg** [MIXTURE] covered 16/20, random fit 0/20, clean=False: One recurring pattern is words or names beginning with “polyc-/polc-” (including polychrome and Polycarp); another is numbers and quantities in reporting, ages, dates, and percentages. These appear to be unrelated patterns.

### mix_L7_N4091_N1020 (planted_mixture, dossier U792)
- intruder P pos/neg: 0.355/0.353 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [MIXTURE] covered 10/20, random fit 1/20, clean=False: Many firings land on pieces of proper names, organizations, or titled places (for example, Rosenzweig, New York, Epic Games, and Golden Gate Bridge). A separate group is phone numbers or other numeric strings; the remaining examples do not support a single shared job.
- **neg** [MIXTURE] covered 6/20, random fit 0/20, clean=False: There are two recurring groups: the ending of “Nehru” in examples 2–8, and marked digits in numeric strings in examples 1, 9, 15, 18, and 19. The other marked words or fragments do not fit either pattern, and the two groups have no clear shared function.

### L10_N5081 (random_neuron, dossier U130)
- intruder P pos/neg: 0.448/0.492 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 19/20, random fit 0/20, clean=True: Fires on names and titles of cultural works or performers—films, television, books, music, and theatre—often in reviews, awards, or arts coverage. The examples span genres and media rather than pointing to one particular kind of work.
- **neg** [ONE] covered 20/20, random fit 0/20, clean=True: Strongly associated with coverage and listings for mainstream commercial film franchises and series, especially superhero, action, and fantasy titles. A couple of examples concern prominent standalone studio films rather than franchise entries.

### L12_N9052 (random_neuron, dossier U716)
- intruder P pos/neg: 0.324/0.249 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 13/20, random fit 0/20, clean=False: The clearest pattern is firing on specific named entities and technical or product labels—such as Apache Harmony, PHPMyAdmin, libc, Atari, Muhammad, CMake, and Urdu. Some examples instead fire on ordinary words or punctuation, so this is a partial pattern rather than a complete account of the pole.
- **neg** [MIXTURE] covered 0/20, random fit 0/20, clean=False: This pole appears to mix at least two patterns: gaming/software terminology and instructions (including game mechanics, platforms, and settings), alongside descriptive vocabulary for decorative objects, artworks, and their imagery. The examples do not provide a convincing shared pattern between those groups.

### L14_N8725 (random_neuron, dossier U818)
- intruder P pos/neg: 0.327/0.435 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 18/20, random fit 0/20, clean=True: Tracks financial or material resources in relation to their available balance: having enough, running low or short, falling into debt, or facing a shortage. The same pattern extends to energy and power supply.
- **neg** [ONE] covered 14/20, random fit 1/20, clean=False: Mostly fires in sports reporting, especially game summaries and player statistics, including pitching, batting, scoring, and defensive plays. The marked fragments often occur in descriptions of those actions or stats.

### L15_N2289 (random_neuron, dossier U559)
- intruder P pos/neg: 0.299/0.353 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 15/20, random fit 5/20, clean=False: Most examples are conversational or public-facing prose that adds evaluation, emphasis, or a personal aside; the topics vary widely, so this looks more like a loose style pattern than a topic detector.
- **neg** [NONE] covered 6/20, random fit 4/20, clean=False: The examples are too varied to support a clear shared job: firings occur on negation and contrast words, but also on ordinary grammatical words in unrelated passages.

### L16_N10008 (random_neuron, dossier U990)
- intruder P pos/neg: 0.304/0.427 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 16/20, random fit 3/20, clean=True: Mostly fires on ordinary words that describe concrete actions, choices, or operations: dialogue and action in narrative, and procedural or quantitative language in puzzle, investing, and game-mechanics text. The shared pattern is operational language rather than a particular subject area.
- **neg** [TWO_RELATED] covered 15/20, random fit 0/20, clean=False: Often fires on franchise-specific names or terms and on abstract plot or lore vocabulary—such as prophecy, destiny, and imagined identity—in fiction summaries and fan discussions. These are related as references to a fictional universe and its lore, though a few firings are just generic grammatical tokens.

### L16_N7735 (random_neuron, dossier U928)
- intruder P pos/neg: 0.412/0.311 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 19/20, random fit 1/20, clean=True: Product descriptions and reviews that explain device or software features, operating controls, settings, modes, or performance. This includes musical equipment and other consumer or technical products.
- **neg** [TWO_RELATED] covered 14/20, random fit 0/20, clean=False: Mostly website privacy, cookie, and terms notices about consent and managing preferences; a related set describes configuring software or service settings. The marked tokens often fall on the wording of a user choice or configuration action.

### L17_N7113 (random_neuron, dossier U925)
- intruder P pos/neg: 0.311/0.245 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [NONE] covered 8/20, random fit 3/20, clean=False: The marked tokens span common function words, content words, and unrelated grammatical roles. A few examples concern change or consequences, but that is too broad to explain the rest reliably.
- **neg** [NONE] covered 0/20, random fit 0/20, clean=False: The firings occur on varied function words, punctuation, names, and content words across unrelated subjects; no consistent lexical, syntactic, or discourse pattern is evident.

### L21_N4969 (random_neuron, dossier U448)
- intruder P pos/neg: 0.227/0.259 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 10/20, random fit 0/20, clean=False: The clearest pattern is language expressing partiality or a portion: “partly” and the repeated disclaimer phrase “comes in part from.”
- **neg** [ONE] covered 14/20, random fit 3/20, clean=False: Most firings are on numerals or number-like expressions: counts, dimensions, dates, and ordinals, as well as “one” used to identify a single item or option.

### L21_N6890 (random_neuron, dossier U814)
- intruder P pos/neg: 0.271/0.302 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [NONE] covered 0/20, random fit 0/20, clean=False: The marked tokens span unrelated topics and roles—page navigation, recipes, technical specifications, academic text, and sports—with no consistent shared feature apparent from the contexts.
- **neg** [NONE] covered 2/20, random fit 4/20, clean=False: These firings likewise occur across unrelated prose, lists, recipes, medical information, and punctuation; the marked positions do not suggest a stable shared pattern.

### L29_N5866 (random_neuron, dossier U967)
- intruder P pos/neg: 0.39/0.23 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 20/20, random fit 0/20, clean=True: The unit tracks suspension, revocation, confiscation, or other removal of a license, right, privilege, or resource, often as a legal penalty. The examples include both explicit license consequences and broader “take away” language.
- **neg** [MIXTURE] covered 7/20, random fit 1/20, clean=False: There are two recurring patterns: disjunctive “or” in employment or regulatory clauses, and number fragments or quantities (such as years, games, scores, and dates). The other marked fragments do not form a clear shared pattern.

### L29_N7253 (random_neuron, dossier U452)
- intruder P pos/neg: 0.343/0.459 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [TWO_RELATED] covered 6/20, random fit 0/20, clean=False: The strongest pattern is quantitative economic analysis, especially pricing, costs, and market measures (1–5, 8, 15). A related pattern is reporting empirical data, outcomes, or study results, often as evidence for a conclusion (6, 10–12, 16–17).
- **neg** [ONE] covered 11/20, random fit 0/20, clean=False: This pole often appears in formal institutional, legal, or public-policy language that lays out rules, criteria, plans, positions, or decisions—frequently in formulations such as “set out” or “set down.”

### L31_N2281 (random_neuron, dossier U144)
- intruder P pos/neg: 0.642/0.232 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [TWO_RELATED] covered 15/20, random fit 0/20, clean=False: The positive pole tracks “taste” in both its literal culinary sense (seasoning something to taste) and its figurative sense of aesthetic or musical preference. These are connected by the familiar extension of taste from palate to judgment and preference.
- **neg** [NONE] covered 4/20, random fit 4/20, clean=False: There is no coherent semantic pattern across these firings; they land on unrelated function words, proper-name fragments, and code or markup tokens.

### L31_N4267 (random_neuron, dossier U847)
- intruder P pos/neg: 0.648/0.27 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 18/20, random fit 0/20, clean=True: The clearest pattern is an orthographic one: firings often land on an initial “L” in a name, place, or word, with no shared topic across those contexts. A few firings in PGP signature blocks and a year do not fit that pattern.
- **neg** [ONE] covered 15/20, random fit 2/20, clean=False: Many examples use language assessing applicability or likely effect: whether a position interests someone, a program fits a budget, something appeals, or a claim may hold. This is a broad evaluative/suitability pattern, often expressed with modal or conditional wording.

### L31_N7961 (random_neuron, dossier U926)
- intruder P pos/neg: 0.488/0.214 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 20/20, random fit 0/20, clean=True: Keyword-stuffed SEO and resource-page copy: repeated search phrases for worksheets, resume formats, and other downloadable or instructional resources, often mixed with generic introductions and title-like keyword lists.
- **neg** [ONE] covered 18/20, random fit 11/20, clean=False: The surrounding text is mostly content-bearing article, review, news, or educational prose, including quotations and named entities, rather than repeated search-keyword titles. The firing targets themselves vary, so the shared pattern is mainly the contrast in surrounding text style.

### L33_N1649 (random_neuron, dossier U227)
- intruder P pos/neg: 0.207/0.258 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 5/20, random fit 0/20, clean=False: A small cluster fires on pieces of specialized or compound vocabulary, such as “Heterogeneous,” “Wasserspiel,” “zonal,” and “vertebra.” This is only a tentative token-level pattern; the other firings do not support a consistent semantic job.
- **neg** [ONE] covered 5/20, random fit 0/20, clean=False: Several firings land on emphatic or intensifying adverbs—“totally,” “absolutely,” and “constantly.” This fits only a subset; the rest look unrelated.

### L34_N3578 (random_neuron, dossier U180)
- intruder P pos/neg: 0.404/0.271 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 13/20, random fit 0/20, clean=False: This pole is associated with physical surfaces as planes or interfaces: where something rests, is exposed, or is measured, and how that surface is shaped or prepared. The examples range from material interfaces to ordinary horizontal or level surfaces.
- **neg** [MIXTURE] covered 12/20, random fit 0/20, clean=False: This pole is a mixture of surface-related wording (often plural or in compounds such as “surface-mount”) and unrelated-looking fragments in words beginning with “res-” or “sur-” (such as “resurfaced,” “resound,” and “surgical”). The text does not support one consistent semantic job for these patterns.

### L6_N4242 (random_neuron, dossier U859)
- intruder P pos/neg: 0.564/0.509 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 18/20, random fit 1/20, clean=True: The marked spans fall inside proper names or named titles—mostly people, but also fictional characters, organizations, places, and a work title. The unit appears to track name-like text rather than one particular topic.
- **neg** [ONE] covered 20/20, random fit 0/20, clean=True: The marked word is “time” in temporal phrases, especially “just in time” and “point-in-time.” These uses refer either to something happening at a timely moment or to a specific instant/state in a technical context.

### L7_N11206 (random_neuron, dossier U204)
- intruder P pos/neg: 0.506/0.378 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 17/20, random fit 1/20, clean=True: Mostly reader-facing cues that flag a caveat, disclaimer, or point to keep in mind (for example, “Note that” and “Bear in mind that”). A few are direct prompts or appeals to the reader, such as “Learn more,” “How are y’all feeling?” and “I urge you.”
- **neg** [ONE] covered 18/20, random fit 0/20, clean=True: Scholarly publication and research-record text: journal titles and citations, repository or bibliographic fields, and academic abstracts or student research sections. The examples vary in topic, but share this formal academic-document context.

### L8_N10131 (random_neuron, dossier U704)
- intruder P pos/neg: 0.394/0.329 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 17/20, random fit 0/20, clean=True: Academic research-publication framing: prose or headings that identify a paper or study and state its purpose, scope, method, or findings, often with “This paper/study…”.
- **neg** [TWO_RELATED] covered 20/20, random fit 2/20, clean=False: Ancillary web-page boilerplate and identifiers: business or site names, domains, and page metadata such as dates and copyright ranges, rather than the main subject prose.

### L8_N11957 (random_neuron, dossier U532)
- intruder P pos/neg: 0.341/0.306 | top-40 firings next to the phrase pos/neg: None/None | double-negative firings removed: 0
- **pos** [ONE] covered 12/20, random fit 0/20, clean=False: Mostly promotional descriptions of games and apps, emphasizing gameplay, characters, features, or player actions. A few examples involve related entertainment products or themed settings rather than games themselves.
- **neg** [TWO_RELATED] covered 19/20, random fit 5/20, clean=False: Factual passages containing identifying or quantitative details: dates and years, measurements, amounts, credentials, citations, or named places and people. Firings often fall on the number or punctuation in those details.
