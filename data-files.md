# Data files

What each labelled-ticket file contains, and what the two Laya result files contain. The wordings were written before any search. The scores were produced later by `noul_matrix.py` and `noul_correlation.py` with `convaiinnovations/laya` on CPU.

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

Ten candidate sentences for each action. `variants` is an object with keys `auto_reply`, `escalate_to_human`, and `ignore`. Each entry is `{ "id", "text" }`. The sentences are different hypotheses about the boundary, kept to one short sentence because Laya keeps 48 tokens of an option.

`a0`, `e0`, and `i0` are the sentences the demo started with. `laya_demo.py` now uses `a1`, `e3`, and `i9`. The `notes` field in this file still describes the original three. `authored_by` records that the sentences were written ahead of the search. `tune_criteria.py` reads `variants` and does not generate text.

## `data/messages.csv`

The fifty messages as one row each. Header: `id`, `label`, `text`. Same ids, labels, and text as `customer_messages.json`. The policy sentences are not in this file. Fields that contain a comma are quoted.

## `data/criteria.csv`

The thirty wordings as one row each. Header: `id`, `action`, `text`. Order is `a0`–`a9`, then `e0`–`e9`, then `i0`–`i9`. Same ids, actions, and text as `criteria_variants.json`. The notes are not in this file.

## `noul_matrix.json`

The 50 by 30 matrix of noul scores, plus the text needed to read it without opening the other files. Produced by `noul_matrix.py` in one batched call, 1,500 scores, 466.6 seconds of model time after the weights had loaded.

| Field | Contents |
| --- | --- |
| `model` | `convaiinnovations/laya` |
| `instruction` | The yes/no question asked for every wording. |
| `false_text` | The false option, the same on every wording. |
| `noul_meaning` | One sentence: the cell is P(the wording matches the message). Row is a message, column is a wording. |
| `seconds` | Model time for the batch, 466.6. |
| `shape` | `[50, 30]` |
| `question_ids` | Column order: `a0` … `a9`, `e0` … `e9`, `i0` … `i9`. |
| `questions` | Those thirty ids with `action` and `text`. |
| `message_ids` | Row order: `m01` … `m50`. |
| `labels` | The true label of each message, in the same order as `message_ids`. |
| `matrix` | Fifty rows of thirty numbers. `matrix[i][j]` is the noul of `question_ids[j]` on `message_ids[i]`. |

`noul_matrix.csv` is the same matrix with a header row. Its columns are `message_id`, `label`, then the thirty wording ids. `noul_correlation.py` reads that CSV.

## `noul_correlation.csv`

Pearson correlation of the thirty wording columns in `noul_matrix.csv`, taken across the fifty messages. Thirty rows and thirty columns of coefficients, written to four decimal places.

The first column is `id`. The header is `id`, then `a0` … `i9` in the same order as the noul matrix. Row `a0`, column `a1` is the correlation of those two wordings. The diagonal is 1, because each wording is correlated with itself. A positive value means the two sentences tend to score high on the same messages and low on the same messages. A negative value means they move in opposite directions.
