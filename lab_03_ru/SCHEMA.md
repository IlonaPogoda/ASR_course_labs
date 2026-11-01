# Схема потока данных — лабораторная работа 03

```text
data/toy_alignment_costs.json ──► viterbi_align ──► (path, total_cost) ──┬─► viterbi_trellis.svg
                                                                          ├─► alignment_path.csv
                                                                          └─► summary.json
data/toy_lm_corpus.txt ──► BigramLM.train ──► lm
data/toy_candidate_scores.json ──► lm.sentence_logprob(transcript) ──► candidate["lm_log_score"]
                                    └─► bigram_probability(prev, word)   (для каждой пары, включая </s>)
candidates + weight ∈ {0, 0.5, 1.0} ──► rank_candidates ──► отсортированный список ──► candidate_scores.csv, summary.json
```

## `scripts/run_lab_03.py`

| Функция | Вход | Выход |
|---|---|---|
| `run_lab(data_dir, output_dir, weights=(0.0, 0.5, 1.0))` | пути к `data/` и `outputs/`, веса LM | `dict` summary; побочно пишет SVG, два CSV и `summary.json` |
| `render_trellis_svg(costs, state_labels, frame_labels, path, output_path)` | матрица `costs[t][s]`, подписи, лучший путь (индексы состояний по кадрам) | `None`; записывает SVG-файл |
| `main()` | аргументы `--data-dir`, `--output-dir` | код возврата `0`; печатает стоимость выравнивания |

## `scripts/viterbi_alignment.py`

| Функция | Вход | Выход |
|---|---|---|
| `viterbi_align(costs, *, start_state=0, final_state=None)` | `costs[t][s]` — 6 кадров × 5 состояний, стоимости эмиссии (−log p, меньше — лучше); `start_state=0`, `final_state=4` | `(path, total_cost)`: `list[int]` длины 6 (состояние на каждом кадре) и `float` — `dp[-1][final_state]` |

Внутри: `dp[t][s]` — накопленная стоимость (инициализирована `inf`, `dp[0][start] = costs[0][start]`), `back[t][s]` — индекс предшественника.
**Пропуски (цикл по `t ≥ 1`, `s`):**
- `choices` — кандидаты-предшественники `(s, dp[t-1][s])` («остаться») и, если `s > start_state`, `(s-1, dp[t-1][s-1])` («перейти»); «остаться» первым, чтобы `min` предпочёл его при равенстве;
- `previous_state, previous_cost` — `min(choices)` по стоимости;
- `dp[t][s] = previous_cost + costs[t][s]`;
- `back[t][s] = previous_state`.

## `scripts/bigram_lm.py`

| Функция | Вход | Выход |
|---|---|---|
| `tokenize(text)` | строка | `list[str]` — слова в нижнем регистре |
| `BigramLM.train(sentences, *, k=0.5)` | строки корпуса, `k > 0` | `BigramLM` с полями `k`, `vocabulary` (слова корпуса + `<unk>` + `</s>`, без `<s>`), `bigram_counts[(h, w)]`, `context_counts[h]` |
| `_context(word)` | слово-контекст | `<s>`, само слово или `<unk>` |
| `_target(word)` | предсказываемое слово | `</s>`, само слово или `<unk>` |
| `bigram_probability(previous, word)` | предыдущее и текущее слово | `float` — `P(w \| h)` со сглаживанием add-k |
| `sentence_logprob(sentence)` | строка-гипотеза | `float` — сумма `log10 P` по всем биграммам от `<s>` до `</s>` |

**Пропуски (`bigram_probability`):**
- `numerator = bigram_counts[(previous, word)] + k`;
- `denominator = context_counts[previous] + k * vocab_size`.

## `scripts/combine_candidate_scores.py`

| Функция | Вход | Выход |
|---|---|---|
| `rank_candidates(candidates, *, weight=1.0)` | список словарей с `acoustic_log_score` и `lm_log_score`; вес `λ ≥ 0` | новый список копий кандидатов с полем `combined_log_score`, отсортированный по убыванию |

**Пропуск:** `row["combined_log_score"] = acoustic + weight * language`.
