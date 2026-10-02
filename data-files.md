# Data files

What each file in `data/` contains, and what the two Laya result files contain. The short backup wordings were written before any search. The noul-length `text` was written later. `noul_matrix.csv` and `noul_matrix.json` were rescored from that longer text. `noul_matrix.backup.csv` and `noul_matrix.backup.json` keep the earlier short-wording scores. `noul_correlation.csv` was computed from the short-wording matrix. The cost matrix is the one `rated_score.py` uses.

Wording ids run `a0`–`a9` (auto_reply), `e0`–`e9` (escalate_to_human), and `i0`–`i9` (ignore). Message ids run `m01`–`m50`.

A cell in the noul files is Laya's `noul`: the probability that the wording matches that message. It is not `answer_confidence`, and the thirty columns do not compete with each other, so a row does not sum to 1. Each wording was its own yes/no question. The true option was the wording's sentence. The false option, on every question, was "This description does not match the customer message." The instruction was "Does this description match the customer message?"

## `data/customer_messages.json`

The labelled set the search and the noul matrix both read. Fifty customer messages, each with `id`, `label`, and `text`.

| Label | Ids | Count | What the label means |
| --- | --- | ---: | --- |
| `auto_reply` | m01–m17 | 17 | A documented how-to. Nothing has failed, and the sender is not asking for money, a replacement, or a security decision. |
| `escalate_to_human` | m18–m34 | 17 | A person has to act: a fault, a damaged or wrong or missing order, a refund, a cancellation, a billing correction, safety, legal, or an account compromise. |
| `ignore` | m35–m50 | 16 | Not a customer asking for help with their own order, account, or device. |

`policy` holds one sentence per label, the rule used when the messages were labelled. `authored_by` records that the messages were written ahead of the search. `tune_criteria.py` reads `messages` and does not generate text.

## `data/criteria_variants.json`

Ten candidate sentences for each action. `variants` is an object with keys `auto_reply`, `escalate_to_human`, and `ignore`. Each entry is `{ "id", "text", "backup" }`. The sentences are different hypotheses about the boundary.

`text` is the noul wording. Laya keeps 48 tokens of a rendered option, for a choice question and for a noul question. A noul true option is rendered as `true: ` plus the sentence, and that prefix is 3 tokens, so the sentence can use the other 45. Each `text` was measured with the Laya tokenizer: ` true: ` plus the sentence is 45 to 48 tokens, so the option is not cut. `backup` is the original short sentence. A choice option spends more of the same 48 tokens on the action label (`escalate_to_human:` is 8 tokens), which is why those lines were shorter.

`laya_demo.py` still uses the short `a1`, `e3`, and `i9` sentences. `tune_criteria.py` reads `text`, so a choice search on this file would truncate the longer sentences. `authored_by` records that the original sentences were written ahead of the search.

## `data/messages.csv`

The fifty messages as one row each. Header: `id`, `label`, `text`. Same ids, labels, and text as `customer_messages.json`. The policy sentences are not in this file. Fields that contain a comma are quoted.

## `data/criteria.csv`

The thirty wordings as one row each. Header: `id`, `action`, `text`, `backup`. Order is `a0`–`a9`, then `e0`–`e9`, then `i0`–`i9`. Same ids, actions, text, and backup sentences as `criteria_variants.json`. The notes are not in this file. Fields that contain a comma are quoted.

## `data/cost_matrix.csv`

The cost of each mistake, as used by `rated_score.py`. Three rows. The first column is `true_label`. The header columns are the action taken: `auto_reply`, `escalate_to_human`, `ignore`. A cell is the cost of taking that action when the row's label is the true one. Lower is better. The diagonal is 0.

| True label | auto_reply | escalate_to_human | ignore |
| --- | ---: | ---: | ---: |
| auto_reply | 0 | 1 | 2 |
| escalate_to_human | 1 | 0 | 3 |
| ignore | 2 | 1 | 0 |

Ignoring a real case costs 3. An automatic reply on that same case costs 1, because the customer still gets an answer. Escalating a how-to or a non-ticket costs 1. An automatic reply to a non-ticket costs 2. `rated_score.py` treats Laya's probabilities as the chance each label is true and picks the action with the lowest expected cost.

## `noul_matrix.json`

The 50 by 30 matrix of noul scores, plus the text needed to read it without opening the other files. Produced by `noul_matrix.py` in one batched call, 1,500 scores, 478.4 seconds of model time after the weights had loaded. The question text stored here is the current `text` field, the noul-length wording.

| Field | Contents |
| --- | --- |
| `model` | `convaiinnovations/laya` |
| `instruction` | The yes/no question asked for every wording. |
| `false_text` | The false option, the same on every wording. |
| `noul_meaning` | One sentence: the cell is P(the wording matches the message). Row is a message, column is a wording. |
| `seconds` | Model time for the batch, 478.4. |
| `shape` | `[50, 30]` |
| `question_ids` | Column order: `a0` … `a9`, `e0` … `e9`, `i0` … `i9`. |
| `questions` | Those thirty ids with `action` and `text`. |
| `message_ids` | Row order: `m01` … `m50`. |
| `labels` | The true label of each message, in the same order as `message_ids`. |
| `matrix` | Fifty rows of thirty numbers. `matrix[i][j]` is the noul of `question_ids[j]` on `message_ids[i]`. |

`noul_matrix.csv` is the same matrix with a header row. Its columns are `message_id`, `label`, then the thirty wording ids. `noul_correlation.py` reads that CSV.

`noul_matrix.backup.csv` and `noul_matrix.backup.json` are the previous run, scored from the short `backup` sentences. That run took 466.6 seconds. `m01` / `a0` is 0.3844 there and 0.4324 in the current file.

## `noul_correlation.csv`

Pearson correlation of the thirty wording columns in the short-wording matrix, taken across the fifty messages. Thirty rows and thirty columns of coefficients, written to four decimal places. It was computed before the rescore, so it describes `noul_matrix.backup.csv`, not the current `noul_matrix.csv`.

The first column is `id`. The header is `id`, then `a0` … `i9` in the same order as the noul matrix. Row `a0`, column `a1` is the correlation of those two wordings. The diagonal is 1, because each wording is correlated with itself. A positive value means the two sentences tend to score high on the same messages and low on the same messages. A negative value means they move in opposite directions.
