# Case studies: Baseline vs TTA

## Case 1: `2hop__498251_126089`

**Question:** Who was in charge of the area where Laren is located?

**Gold answer:** `['Johan Remkes']`

**Baseline answer:** `Lars Levi Laestadius`

**TTA answer:** `Johan Remkes`

| Metric | Baseline | TTA | Delta |
|---|---:|---:|---:|
| em | 0.00 | 100.00 | +100.00 |
| f1 | 0.00 | 100.00 | +100.00 |
| precision | 0.00 | 100.00 | +100.00 |
| recall | 0.00 | 100.00 | +100.00 |

**Gold evidence:**

- `Laren, North Holland`: Laren () is a town and municipality in the province of North Holland, Netherlands. Located in the Gooi region, it is the oldest town in the area. It is one of the most affluent in the Netherlands, along with its neighbour Blaricum.
- `North Holland`: The capital and seat of the provincial government is Haarlem, and the province's largest city is the Netherlands' capital Amsterdam. The King's Commissioner of North Holland is Johan Remkes, serving since 2010. There are 51 municipalities and three (including parts of) water boards in the province.

**Retrieval trajectory:**

### Iteration 1

| Rank | Baseline top docs | TTA top docs |
|---:|---|---|
| 1 | Lars Levi Laestadius | Laren, North Holland |
| 2 | Amundsen's South Pole expedition | North Holland |
| 3 | Egil Olbjørn | Sleen |
| 4 | Franz Neuhausen | Bangert, North Holland |
| 5 | Christiaan Lans | Well, Limburg |

TTA diagnostics: query_shift=1.2087, l1_steps=1, l1_loss=3.493225574493408, l2_loss=3.030733823776245

### Iteration 2

| Rank | Baseline top docs | TTA top docs |
|---:|---|---|
| 1 | Middle Ages | New Amsterdam |
| 2 | Middle Ages | New York City |
| 3 | Middle Ages | New York City |
| 4 | History of Finland | Claes Gerritszoon Compaen |
| 5 | Near East | Jurriaen Aernoutsz |

TTA diagnostics: query_shift=2.0268, l1_steps=2, l1_loss=1.922850489616394, l2_loss=1.702957034111023


## Case 2: `2hop__557284_160249`

**Question:** Which years did the war with Antigone's spouse occur?

**Gold answer:** `['323–272 BC']`

**Baseline answer:** `The Third Macedonian War, which involved Philip V's son Perseus, occurred from 171 to 168 BCE. The Hellenistic period was a time of great turmoil and upheaval in the ancient world. The death of Alexander the Great in 323 BC led to a series of wars and power struggles among his successors, including the Diadochi. "" "The years of the war with Antigone's spouse are not specified in the provided documents, but the Third Macedonian War, which involved Philip V's son Perseus, occurred from 171 to 168 "" "The years of the war with Antigone's spouse are not specified in the provided documents, but the Third Macedonian War, which involved Philip V's son Perseus, occurred from 171 to 168 "" "The years of the war with Antigone's spouse are not specified in the provided documents, but the Third Macedonian War, which involved Philip V's son Perseus, occurred from 171 to 168 "" "The years of the war with Antigone's spouse are not specified in the provided documents, but the Third Macedonian War, which involved Philip V's son Perseus, occurred from 171 to 168`

**TTA answer:** `323–272 BC`

| Metric | Baseline | TTA | Delta |
|---|---:|---:|---:|
| em | 0.00 | 100.00 | +100.00 |
| f1 | 4.00 | 100.00 | +96.00 |
| precision | 2.00 | 100.00 | +98.00 |
| recall | 50.00 | 100.00 | +50.00 |

**Gold evidence:**

- `Hieronymus of Cardia`: He wrote a history of the Diadochi and their descendants, encompassing the period from the death of Alexander to the war with Pyrrhus (323–272 BC), which is one of the chief authorities used by Diodorus Siculus (xviii.–xx.) and also by Plutarch in his life of Pyrrhus.
- `Ptolemy (son of Pyrrhus)`: Ptolemy (295–272 BC) was the oldest son of king Pyrrhus of Epirus and his first wife Antigone, who probably died in childbirth. He was named in honour of his mother's stepfather, king Ptolemy I Soter of Egypt, who was a benefactor to Pyrrhus in his youth.

**Retrieval trajectory:**

### Iteration 1

| Rank | Baseline top docs | TTA top docs |
|---:|---|---|
| 1 | Northern Seven Years' War | Hellenistic period |
| 2 | Hellenistic period | Antipater II of Macedon |
| 3 | French and Indian War | Hellenistic period |
| 4 | 18th century | Hellenistic period |
| 5 | Trojan War | Hieronymus of Cardia |

TTA diagnostics: query_shift=1.2060, l1_steps=1, l1_loss=3.1584227085113525, l2_loss=2.904989242553711

### Iteration 2

| Rank | Baseline top docs | TTA top docs |
|---:|---|---|
| 1 | Hellenistic period | Phthia of Epirus |
| 2 | Hellenistic period | Neoptolemus |
| 3 | Greece | Seleucus V Philometor |
| 4 | Hellenistic period | Haemon |
| 5 | Roman Republic | Olympias |

TTA diagnostics: query_shift=2.0782, l1_steps=2, l1_loss=3.203315496444702, l2_loss=3.0929949283599854

### Iteration 3

| Rank | Baseline top docs | TTA top docs |
|---:|---|---|
| 1 | Late Middle Ages |  |
| 2 | Hellenistic period |  |
| 3 | 1st century |  |
| 4 | Hellenistic period |  |
| 5 | Modern history |  |
### Iteration 4

| Rank | Baseline top docs | TTA top docs |
|---:|---|---|
| 1 | British Empire |  |
| 2 | Middle Ages |  |
| 3 | Modern history |  |
| 4 | Middle Ages |  |
| 5 |  |  |
### Iteration 5

| Rank | Baseline top docs | TTA top docs |
|---:|---|---|
| 1 | History of India |  |
| 2 | England in the Middle Ages |  |
| 3 | Modern history |  |
| 4 | Middle Ages |  |
| 5 | Egypt |  |
### Iteration 6

| Rank | Baseline top docs | TTA top docs |
|---:|---|---|
| 1 | Pre-Columbian era |  |
| 2 | Middle Ages |  |
| 3 | Middle Ages |  |
| 4 | Middle Ages |  |
| 5 |  |  |
### Iteration 7

| Rank | Baseline top docs | TTA top docs |
|---:|---|---|
| 1 | Middle Ages |  |
| 2 | Late Middle Ages |  |
| 3 | Muslim world |  |
| 4 |  |  |
| 5 |  |  |

## Case 3: `2hop__335899_124498`

**Question:** Who is the father of the performer of Qui de noux deux?

**Gold answer:** `['Louis Chedid']`

**Baseline answer:** `Robert Gall`

**TTA answer:** `Matthieu Chedid`

| Metric | Baseline | TTA | Delta |
|---|---:|---:|---:|
| em | 0.00 | 0.00 | +0.00 |
| f1 | 0.00 | 50.00 | +50.00 |
| precision | 0.00 | 50.00 | +50.00 |
| recall | 0.00 | 50.00 | +50.00 |

**Gold evidence:**

- `Matthieu Chedid`: Matthieu Chedid was born in Boulogne-Billancourt, Hauts-de-Seine, France, the son of French singer Louis Chedid, and the grandson of the Egyptian-born French writer and poet of Lebanese descent Andrée Chedid who has written lyrics for him. His sister is the music video and concert director Émilie Chedid.
- `Qui de nous deux`: Qui de nous deux ? (2003) is an album by French singer-songwriter Matthieu Chedid in his persona as -M-. It is his third full studio album, or the fourth if including the instrumental work Labo M which he had released earlier the same year. The title of the album and its best known track means 'Which of Us Two' reflecting the ongoing contrast betwe

**Retrieval trajectory:**

### Iteration 1

| Rank | Baseline top docs | TTA top docs |
|---:|---|---|
| 1 | Friedrich Wilhelm Quirin von Forcade de Biaix | Qui de nous deux |
| 2 | Jacques Février | Labo M |
| 3 | Robert Gall | Je dis aime |
| 4 | Maurice de Féraudy | Double je |
| 5 | Antoine Forqueray | Keine Macht für Niemand |

TTA diagnostics: query_shift=1.2069, l1_steps=1, l1_loss=3.4791243076324463, l2_loss=2.8159031867980957

### Iteration 2

| Rank | Baseline top docs | TTA top docs |
|---:|---|---|
| 1 | Auguste Léopold Protet | Louis Chedid |
| 2 | Nicolas-Germain Léonard | Matthieu Chedid |
| 3 | Jean Alfonse | Fernand Sardou |
| 4 | Jacques Vallée, Sieur Des Barreaux | Mari Maurstad |
| 5 | Charles-Pierre Colardeau | Aimé Clariond |

TTA diagnostics: query_shift=1.2050, l1_steps=1, l1_loss=3.25785231590271, l2_loss=2.4852652549743652

### Iteration 3

| Rank | Baseline top docs | TTA top docs |
|---:|---|---|
| 1 | John, King of England |  |
| 2 | Alps |  |
| 3 | John, King of England |  |
| 4 | Ernest Doudart de Lagrée |  |
| 5 | Middle Ages |  |
### Iteration 4

| Rank | Baseline top docs | TTA top docs |
|---:|---|---|
| 1 | Napoleon |  |
| 2 | Middle Ages |  |
| 3 | Johannes V. Jensen |  |
| 4 | Charles-Louis Pinson de Ménerville |  |
| 5 | Georges de Rham |  |

## Case 4: `2hop__6735_6733`

**Question:** To what division does the language that comprises the standard pronunciation of Catalan belong?

**Gold answer:** `['Eastern block']`

**Baseline answer:** `Western Romance division`

**TTA answer:** `Eastern Catalan`

| Metric | Baseline | TTA | Delta |
|---|---:|---:|---:|
| em | 0.00 | 0.00 | +0.00 |
| f1 | 0.00 | 50.00 | +50.00 |
| precision | 0.00 | 50.00 | +50.00 |
| recall | 0.00 | 50.00 | +50.00 |

**Gold evidence:**

- `Catalan language`: Western Catalan comprises the two dialects of Northwestern Catalan and Valencian; the Eastern block comprises four dialects: Central Catalan, Balearic, Rossellonese, and Alguerese. Each dialect can be further subdivided in several subdialects.
- `Catalan language`: Central Catalan is considered the standard pronunciation of the language and has the highest number of speakers. It is spoken in the densely populated regions of the Barcelona province, the eastern half of the province of Tarragona, and most of the province of Girona.

**Retrieval trajectory:**

### Iteration 1

| Rank | Baseline top docs | TTA top docs |
|---:|---|---|
| 1 | Catalan language | Catalan language |
| 2 | Catalan language | Catalan language |
| 3 | Catalan language | Catalan language |
| 4 | Catalan language | Catalan language |
| 5 | Catalan language | Catalan language |

TTA diagnostics: query_shift=1.2086, l1_steps=1, l1_loss=2.581479787826538, l2_loss=2.4501116275787354

### Iteration 2

| Rank | Baseline top docs | TTA top docs |
|---:|---|---|
| 1 | Dialect | Catalan language |
| 2 | Dialect | Catalan language |
| 3 | Catalan language | Catalan language |
| 4 | Dialect | Catalan language |
| 5 |  | Catalan language |

TTA diagnostics: query_shift=1.2126, l1_steps=1, l1_loss=2.7951583862304688, l2_loss=1.9688819646835327


## Case 5: `2hop__804417_126089`

**Question:** Who was in charge of the place where Bergen is located?

**Gold answer:** `['Johan Remkes']`

**Baseline answer:** `Governor Jørgen Iversen Dyppel`

**TTA answer:** `The King's Commissioner of North Holland, Johan Remkes`

| Metric | Baseline | TTA | Delta |
|---|---:|---:|---:|
| em | 0.00 | 0.00 | +0.00 |
| f1 | 0.00 | 44.00 | +44.00 |
| precision | 0.00 | 29.00 | +29.00 |
| recall | 0.00 | 100.00 | +100.00 |

**Gold evidence:**

- `Bergen, North Holland`: Bergen () is a municipality and a town in the Netherlands, in the province of North Holland. Its North Sea beaches make it a popular destination for tourists. In 2001, the municipality was formed from a merger of the former municipalities of Egmond, Schoorl, and the smaller community of Bergen proper that had existed since 1811.
- `North Holland`: The capital and seat of the provincial government is Haarlem, and the province's largest city is the Netherlands' capital Amsterdam. The King's Commissioner of North Holland is Johan Remkes, serving since 2010. There are 51 municipalities and three (including parts of) water boards in the province.

**Retrieval trajectory:**

### Iteration 1

| Rank | Baseline top docs | TTA top docs |
|---:|---|---|
| 1 | Bergensposten | Bergen, North Holland |
| 2 | Gamlehaugen | Bergen, North Dakota |
| 3 | Fort Christian | Kreis Bergen |
| 4 | Ole Peter Riis Høegh | Well, Limburg |
| 5 | Troldhaugen | Egmond aan Zee |

TTA diagnostics: query_shift=1.2099, l1_steps=1, l1_loss=2.852482557296753, l2_loss=2.492302417755127

### Iteration 2

| Rank | Baseline top docs | TTA top docs |
|---:|---|---|
| 1 | Middle Ages | Dutch Republic |
| 2 | St. Augustine, Florida | New York City |
| 3 | French colonization of the Americas | Dutch Republic |
| 4 | St. John's, Newfoundland and Labrador | Dutch Republic |
| 5 | History of the United States Virgin Islands | New Amsterdam |

TTA diagnostics: query_shift=2.0065, l1_steps=2, l1_loss=3.77545166015625, l2_loss=3.3489608764648438

