# Combining Laya decisions to cut routing cost

The development set is the 50 labelled messages in `data/customer_messages.json`. The cost of a decision is `data/cost_matrix.csv`. Lower is better. A correct action costs 0. Ignoring a real escalation costs 3. An automatic reply on a non-ticket costs 2. Escalating a how-to or a non-ticket costs 1.

A **Laya call** here is one 3-option `choice` question on one message. Packing two questions into one `predict_batch` still counts as two calls per message, because each question is its own decision.

These numbers were produced by running `convaiinnovations/laya` (package `laya` 0.3.20) on CPU in this environment. They are not estimates. Ollama is not part of the cost: `llama3.1:8b` only drafts a reply after the action is chosen, and this experiment stops at the action. The model was not pulled.

## Prior work

Paul Yardley, “Tuning Laya’s option text against a labelled ticket set”, 29 September 2026. <https://paulyardleyqa.co.uk/blog/laya-criteria-tuning/>

That search is why `laya_demo.py` uses wording `a1` / `e3` / `i9` and routes on the highest probability. On these 50 messages that wording scores 41/50 and costs 19. The original wording `a0` / `e0` / `i0` scores 19/50 and costs 35. The nine misses were stable across nearby wordings. This file treats cost 19 as the figure to beat, and the live run below reproduces it.

Laya keeps the first 48 tokens of each option, rendered as `label: description`. All options share a 192-token head. Eleven or more options use the checkpoint’s `choice:11+` temperature, which this build clamps and treats as uncalibrated. Every call in this experiment has exactly three options, and each option is under the cap. The `choice:11+` warning still prints on load. It does not apply to these calls.

The write-up already measured the following. They were not run again as searches.

| Earlier attempt | What it did | Why it is not repeated here |
| --- | --- | --- |
| (a) All ten wordings joined into one option | 17/50. The text was cut at 48 tokens and the winning sentences sat past the cut. | A longer option is not a second decision. Token audit below refuses a question that would be cut. |
| (b) Ten labels per action, 30 options, summed back into three classes | Not run. It breaks the 192-token head and lands in `choice:11+`. | Same constraint. Not run. |
| (c) Average the probabilities of the nearest strong wordings | Stayed at 40–41/50 with the same nine misses. Those wordings disagree on messages the winner already gets right. | Averaging nearby paraphrases of the same route question does not move the misses. Not run again. |
| (d) A second 3-way question on messages already sent to `auto_reply`, and a re-check of `ignore` | 40/50, 36/50, and 39/50. The second question asked for the same three actions in different words. | A follow-up has to do a different job. The specialist questions below are that attempt. The old second pass was not copied. |
| (e) Lowest expected cost, treating the three probabilities as class chances | 26/50, cost 25. Probabilities sit around 0.3–0.5, so escalation wins the close calls. | Recomputed on this run’s demo probabilities as a control, not as a new rule. Same result: cost 25, 26/50. No extra Laya call. |
| (f) A 70% confidence gate | Correct calls sit around 0.51–0.58, so the gate blocks almost everything. | Recomputed on this run as a control. Cost 33, 17/50, and every message is escalated. One of the 50 clears 70%, and the gate still escalates it because the other 49 do not. No extra Laya call. |

## How to run

From the repo root, with the dependencies in `requirements.txt` (`laya==0.3.20` and a CPU build of PyTorch):

```bash
pip install -r requirements.txt
python experiments/run_all.py
```

`run_all.py` only accepts the original 50 messages, because it fits thresholds. It writes `experiments/results/raw_decisions.json`, `experiments/config.json`, and one JSON per strategy. A second run reuses the saved decisions unless you pass `--fresh`.

On the CPU that produced this log, loading the checkpoint took 7.5 seconds (the weights were fetched at the start of that interval) and one pass over the 50 messages took 6.8 seconds. The blog’s machine takes about 18–20 seconds a pass. Budget that if yours matches the write-up. The winning rule needs one pass. Scoring every rule needs four passes: the demo question twice, then `ownership` and `need` together.

Each strategy can also be scored on its own, against a decisions file that `collect.py` or `apply.py` has already written:

```bash
python experiments/01_baseline.py --messages data/customer_messages.json
python experiments/06_repair_thresholds.py --messages data/customer_messages.json
```

`--messages` accepts the repo JSON or a CSV. The default is `data/customer_messages.json`.

## Questions and token counts

Token counts use the checkpoint tokenizer, with the same rendering as Laya (`label: description`, plus a leading space, plus the MASK token that sits in front of each option). The instruction is `choice question: …`. Nothing was truncated.

| Question | Job | Instruction tokens | Option tokens | Options + MASK tokens | Head cap |
| --- | ---: | ---: | --- | ---: | ---: |
| `demo` | The route, wording `a1` / `e3` / `i9` | 14 | auto_reply 26, escalate_to_human 23, ignore 8 | 60 | 192 |
| `ownership` | Who is writing, relative to this shop | 12 | customer 15, outsider 17, unclear 13 | 48 | 192 |
| `need` | What kind of help the sender needs | 12 | steps 16, staff 19, none 14 | 52 | 192 |

`demo` is the instruction and the three sentences in `laya_demo.py`. `ownership` and `need` are not a second copy of those three route labels. Their sentences follow the labelling policy in `data/customer_messages.json`. They do not quote the nine missed messages. The exact strings are in `experiments/questions.py` and are copied into `experiments/config.json` so a later edit of the script cannot change a held-out run.

Settings shared by every live call: model `convaiinnovations/laya`, `laya` 0.3.20, device `cpu`, shipped temperature (not overridden), batch size 4. After load, `model.training` was already `False`.

## Development results

Raw decisions: `experiments/results/raw_decisions.json`. One row per strategy and message: `experiments/results/per_message.csv`. Summary: `experiments/results/summary.csv`. The frozen fit: `experiments/config.json`.

The demo question was asked twice. Every choice and every probability matched. Repeated sampling of the same prompt cannot change the vote. The repeat was stopped at two calls.

The live demo choices and probabilities also match `criteria_search_results.json` for `a1` / `e3` / `i9` on all 50 messages. The published cost of 19 is reproduced, not assumed.

| Strategy | Tuned? | Cost on the 50 | Leave-one-out cost | Accuracy | Laya calls | Compared with baseline |
| --- | --- | ---: | ---: | --- | ---: | --- |
| `baseline_argmax` | no | **19** | — | 41/50 | 50 | — |
| `repeated_majority` | no | 19 | — | 41/50 | 100 | same decisions, twice the calls |
| `specialist_route` | no | 85 | — | 16/50 | 100 | worse |
| `mapped_majority` | no | 45 | — | 30/50 | 150 | worse |
| `costly_miss_repair` | no | 37 | — | 32/50 | 101 | worse |
| `repair_thresholds` | yes | 17 (frozen rule, reapplied) | **19** | 42/50 frozen, 41/50 leave-one-out | 101 | leave-one-out ties the baseline |

Call counts are what the rule needs, not how many calls the collector spent while scoring every rule from one cache. The collector made 200 calls (two demo passes and one pass that asked both specialist questions, 50 messages each).

### 1. Baseline: highest probability on `a1` / `e3` / `i9`

One call. The action is `choice`, which is the highest of the three probabilities. This is `laya_demo.py`.

Cost 19. Accuracy 41/50. Calls: 50.

| True label | auto_reply | escalate_to_human | ignore |
| --- | ---: | ---: | ---: |
| auto_reply | 14 | 2 | 1 |
| escalate_to_human | 0 | 14 | 3 |
| ignore | 3 | 0 | 13 |

The nine off-diagonal cells are the ones in the write-up: two how-tos escalated (`m02`, `m15`), one how-to ignored (`m05`), three real cases ignored (`m24`, `m32`, `m33`), three non-tickets given an automatic reply (`m36`, `m37`, `m46`).

Controls on the same probabilities, same 50 calls, not candidates:

| Control | Cost | Accuracy | Confusion (rows are the true label) |
| --- | ---: | --- | --- |
| Expected cost | 25 | 26/50 | auto_reply 7/10/0; escalate 0/17/0; ignore 1/13/2 |
| 70% gate, else the top probability | 33 | 17/50 | auto_reply 0/17/0; escalate 0/17/0; ignore 0/16/0 |

Both match the write-up. Expected cost fixes the three dropped cases by escalating almost every close call. The 70% gate sends every message to a human. The one score that clears 70% is `m26`, already an escalation at 0.8525, so the gate never opens another route.

### 2. Repeated majority

Same demo question, two calls, majority vote. A disagreement would have been broken by the pre-specified order escalate, then auto-reply, then ignore (lowest worst-case cost first). There was no disagreement.

Cost 19. Accuracy 41/50. Calls: 100. The rule ties the baseline and spends an extra call on every message, so it loses.

### 3. Specialist route

`ownership` and `need` are asked on every message. A fixed map, with no fitted weight, turns them into a route:

- ownership `outsider`, or need `none` → `ignore`
- otherwise need `staff` → `escalate_to_human`
- otherwise need `steps` and ownership `customer` → `auto_reply`
- otherwise the demo choice

`need` answered `none` on all 50 messages. The map therefore returns `ignore` every time and never reaches the demo. Cost 85. Accuracy 16/50 (the 16 true non-tickets). Calls: 100. The different job was real; the `need` question did not separate anyone.

| True label | auto_reply | escalate_to_human | ignore |
| --- | ---: | ---: | ---: |
| auto_reply | 0 | 0 | 17 |
| escalate_to_human | 0 | 0 | 17 |
| ignore | 0 | 0 | 16 |

### 4. Mapped majority

Three votes: the demo choice, `ignore` when ownership says `outsider`, and a route vote from `need` (`steps` → auto-reply, `staff` → escalate, `none` → ignore). A tie that still includes the demo choice keeps the demo, so the specialists change the route only when they outvote it.

Because `need` is `none` on every message, `ignore` starts with one vote before ownership is considered. Cost 45. Accuracy 30/50. Calls: 150.

| True label | auto_reply | escalate_to_human | ignore |
| --- | ---: | ---: | ---: |
| auto_reply | 4 | 0 | 13 |
| escalate_to_human | 0 | 12 | 5 |
| ignore | 2 | 0 | 14 |

Ten how-tos that the demo got right are flipped to ignore, and two further real cases are ignored. That is the expensive direction. The rule loses.

### 5. Costly-miss repair, no threshold

The demo choice stands unless a specialist flags one of the two expensive patterns. Confidence is not gated (`t_own = 0`, `t_need = 0`): the specialist’s top label is enough.

- Demo `ignore`, ownership `customer`, and need `staff` → `escalate_to_human`
- Demo `auto_reply` and ownership `outsider` → `ignore`
- Demo `escalate_to_human` is never changed. Turning an escalation into ignore is the cost-3 mistake.

`need` never says `staff`, so the first flip never happens. The second flip does. Ownership says `outsider` often enough, including on real how-tos, that ten correct automatic replies become ignores. Cost 37. Accuracy 32/50. Calls: 101 (the demo on all 50, both specialists when the demo said ignore, ownership alone when it said auto-reply).

| True label | auto_reply | escalate_to_human | ignore |
| --- | ---: | ---: | ---: |
| auto_reply | 4 | 2 | 11 |
| escalate_to_human | 0 | 14 | 3 |
| ignore | 2 | 0 | 14 |

### 6. Repair with frozen thresholds

Same flips as strategy 5, but each specialist has to clear a confidence threshold. The grid is 0.00 to 1.00 in steps of 0.05, plus 1.01, which cannot fire because `answer_confidence` is at most 1. The fit minimises total cost on all 50. Ties take the higher thresholds, so the rule stays quiet unless firing actually helps.

Fit on all 50: `t_own = 0.60`, `t_need = 1.01`. The need-threshold is the off switch. That is forced by the data: `need` is `none` everywhere, so no value of `t_need` changes a decision, and the tie-break picks the highest one.

Reapplied to the same 50, the frozen rule changes one message. `m37` (true label `ignore`, demo `auto_reply` at 0.4765) has ownership `outsider` at 0.6454, which clears 0.60. The prediction becomes `ignore`. Cost goes from 2 to 0. No other message moves. Frozen cost 17. Frozen accuracy 42/50. Calls: 101.

| True label | auto_reply | escalate_to_human | ignore |
| --- | ---: | ---: | ---: |
| auto_reply | 14 | 2 | 1 |
| escalate_to_human | 0 | 14 | 3 |
| ignore | 2 | 0 | 14 |

Leave-one-out tells a different story. For each message the thresholds are fit on the other 49 only, then applied to the message that was held out.

- Holding out any message except `m37` still leaves `m37` in the fit, so the fold chooses `0.60 / 1.01`. The held-out message is not `m37`, so the rule does not change it.
- Holding out `m37` leaves a training set where firing never helps. The tie-break chooses `1.01 / 1.01`. `m37` is then scored with a rule that does not fire, and it stays a wrong automatic reply.

Forty-nine folds chose `0.60 / 1.01`. One fold chose `1.01 / 1.01`. Leave-one-out cost is 19. Leave-one-out accuracy is 41/50. The two-point saving exists only when the threshold is allowed to see `m37`.

## Overfitting

The wording `a1` / `e3` / `i9` was chosen on these 50 messages. Cost 19 is that wording’s training score. A fresh file, labelled the same way, is what would show whether the wording holds. The blog says this, and it is still true.

The specialist sentences were written from the labelling policy before this run, not picked off the criteria leaderboard, and they were not rewritten after the run. `need` answering `none` everywhere is a failed question, left as it is. Changing the sentence now would be fitting text to these 50.

`repair_thresholds` is the only rule with fitted numbers. Two thresholds on 50 messages is enough to memorise a single flip, and that is what happened. The number used to rank it is the leave-one-out cost, 19, not the reapplied cost, 17. The config still stores `t_own = 0.60` and `t_need = 1.01`, because a later file should be scored with the rule that was fit on all of the development data, not with a threshold refit on that later file. Whether 0.60 is anything more than “catch `m37`” is exactly what the held-out file is for. It is not assumed here.

Untuned rules have no free parameter. Their costs above are the whole measurement. They are still measurements on the set the demo wording was chosen on, so a win on a different file would be the confirmation. None of them won here.

Ranking rule, in `fit_config.py`: lowest comparison cost; a tuned rule contributes its leave-one-out cost; ties prefer an untuned rule, then fewer Laya calls. `repair_thresholds` ties the baseline at 19 and is tuned, and it uses more calls, so it loses the tie.

## Conclusion

The lowest cost we can claim on the development set is **19**, from **`baseline_argmax`**: one Laya call per message, the tuned wording, the highest probability. It needs **50** calls.

The frozen repair rule scores 17 when the threshold is reapplied to the messages it was fit on, and 19 under leave-one-out. It is not the winner. Repeated calls of the same prompt are identical, so a majority over samples cannot help. The two specialist questions, which were the attempt at a second decision with a different job, do not beat the single tuned wording. `need` does not vary. `ownership` is not sharp enough to repair the nine misses without either doing nothing (`t_own = 0.60` under leave-one-out) or turning correct how-tos into ignores (no threshold).

## How to test on a new message set

Do not refit. `experiments/fit_config.py` and `experiments/run_all.py` refuse any file other than the original 50. `experiments/apply.py` reads `experiments/config.json` and does not search thresholds, weights, or question text.

The new file can be CSV or JSON. CSV columns are `id`, `label`, `text`. A field that contains a comma is quoted. `label` is one of `auto_reply`, `escalate_to_human`, `ignore`.

```csv
id,label,text
m01,auto_reply,How do I reset my password? I can't remember it.
```

JSON is either a list of objects with those three fields, or `{"messages": [ ... ]}` as in `data/customer_messages.json`.

Score the winning strategy. This asks only the demo question (one Laya call per message) and does not refit:

```bash
python experiments/apply.py \
  --messages path/to/new_messages.csv \
  --config experiments/config.json \
  --out experiments/results/heldout
```

Score every frozen strategy, which is the run that fills the table below. This also asks `ownership` and `need`, and it asks the demo question twice. Thresholds stay at the saved `t_own = 0.60` and `t_need = 1.01`.

```bash
python experiments/apply.py \
  --messages path/to/new_messages.csv \
  --config experiments/config.json \
  --all \
  --out experiments/results/heldout
```

Expect about 7 seconds to load the checkpoint if it is already downloaded, then about 7 seconds per 50-message pass on a CPU like the one that wrote this log. On a machine that matches the blog, budget about 18–20 seconds per pass. The winner is one pass. `--all` is four passes. The first run on a new machine also downloads `convaiinnovations/laya` from Hugging Face. No API key and no Ollama model are required.

If `apply.py` prints that the file is the development set, the path still points at the original 50. That reprint is not a held-out result.

## Held-out results

Not yet run. There is no second labelled file in the repo. The cells below are empty on purpose. Do not copy the development costs into them.

| Strategy | Total cost | Accuracy | Laya calls | Notes |
| --- | --- | --- | --- | --- |
| `baseline_argmax` (frozen winner) | not yet run | not yet run | not yet run | One demo call per message. No fitted threshold. |
| `repeated_majority` | not yet run | not yet run | not yet run | Two demo calls. On the development set they were identical. |
| `specialist_route` | not yet run | not yet run | not yet run | Untuned map. No refit. |
| `mapped_majority` | not yet run | not yet run | not yet run | Untuned vote. No refit. |
| `costly_miss_repair` | not yet run | not yet run | not yet run | Thresholds fixed at 0. No refit. |
| `repair_thresholds` | not yet run | not yet run | not yet run | Must use frozen `t_own = 0.60`, `t_need = 1.01`. Must not refit. |
