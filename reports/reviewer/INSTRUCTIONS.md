# Reviewer check: instructions (about 30 minutes)

Thank you for helping. This is the blind reviewer check from our plan (§11.1): two teammates who did
not build the scoring each rate the same 40 Bitcoin transactions. Your answers are reported as
judgement, never as ground truth.

## What to do

1. Open `sheet.csv` (Excel is fine). Each row is one transaction of one address.
2. For every row, compare the transaction with **that address's usual values** in the same row, and
   fill the `worth_a_look` column with **yes** or **no**:
   - **yes**: something about it is clearly different from what this address usually does, and an
     analyst could reasonably want to look at it;
   - **no**: it looks like the address's normal activity.
3. Optionally write a few words in `note` (for example "amount 30x the usual").
4. Save the file as `ratings_<your initials>.csv` (for example `ratings_mkn.csv`) and send it back.

Please work alone and don't discuss rows with the other reviewer until you have both sent your files.

## The columns

| Column | Meaning |
|---|---|
| `amount_btc` / `usual_amount_btc` | How much moved, and the address's usual amount in the same direction |
| `tx_in_previous_7_days` / `usual_tx_in_7_days` | How busy the previous week was, and how busy a week usually is |
| `hours_since_previous` / `usual_hours_between` | The gap before this transaction, and the usual gap |
| `direction` | Received or sent, from this address's point of view |
| `inputs`, `outputs`, `fee_rate_sat_per_vb` | The transaction's shape and fee |
| `explorer_link` | The transaction on a public block explorer, if you want to see it |

## Please keep in mind

- Some rows were flagged by our method and some were not, shuffled; the sheet does not say which.
- "Worth a look" means **unusual for this address**. It never means that anyone did anything wrong:
  unusual activity requires investigation; it does not indicate wrongdoing.
- Every row needs a yes or a no; a blank row cannot be scored.
