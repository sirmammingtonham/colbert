# colbert
reimplementation of colbert

## Description
I implemented my own version of ColBERT from scratch to get a better understanding of the smaller details, which are implemented in this repo.

However training would have taken a long time, and the main project was to implement ColBERT style late interaction in the LUAR repo anyway. 

Branch with late interaction code can be viewed here: https://github.com/srush/luar/tree/latest_interaction

## Experiment Setup
Trying to use late interaction with luar model.
* Change as little code as possible to implement late interaction
  * Removed pooling layers in the model
  * Changed distance metric in the loss function and validation code to use late interaction/maxsim instead of cosine similarity
  * To make late interaction distances bidirectional I calculate the score in both directions and average them
* Vary locations where we calculate maxsim (over docs, lengths, etc.)

### Original model:
**Input/output:** (batch, docs, length) -> (batch, hidden) \
**Model changes:** meanpool over lengths, attention between docs, maxpool over docs \
**Loss distance:** loss = (batch, embedding) @ (batch, embedding).T 

### Maxsim over docs:
**Input/output:** (batch, docs, length) -> (batch, docs, hidden) \
**Model changes:** meanpool over lengths, attention between docs, remove maxpool over docs \
**Loss distance:** scores = einsum("bwd,ped->bpwe"), \
loss = (scores.max(-1).sum(-1) + scores.max(-2).sum(-1)) / 2 

### Maxsim over lengths:
**Input/output:** (batch, docs, length) -> (batch, length, hidden) \
**Model changes:** remove meanpool over lengths, attention between docs, maxpool over doc \
**Loss distance:** scores = einsum("bwd,ped->bpwe"), \
loss = (scores.max(-1).sum(-1) + scores.max(-2).sum(-1)) / 2 

### Maxsim over lengths and docs:
**Input/output:** (batch, docs, length) -> (batch, docs, length, hidden) \
**Model changes:** remove meanpool over lengths, remove attention between docs, remove maxpool over docs \
**Loss distance:** scores = einsum("beld,pwsd->bpewls"), \
loss = (scores.max(-1).mean(-1).max(-1).mean(-1) + scores.max(-2).mean(-1).max(-2).mean(-1)) / 2 \
** Had to use mean instead of sum here since the training seemed unstable with sum (loss > 100)

### Reimplimentation:
Sanity check to reimpliment original model using my changes to the loss distance and the model. Basically wanted to make sure that my changes didn't break anything if I use the same exact loss function and model. Does the same computation as the original model but using the outputs as maxsim over docs (batch, docs, hidden).
I compute maxpooling in the distance function, and reimpliment cosine similarity rather than using pytorch metric learning's implementation. 

Performs identical to the original (actually performs slightly better, difference probably because the reimplimentation uses a linear layer before maxpool, where original uses a maxpool then linear).

### Other experiments:
Ran a lot of other experiments removing attention, doing max then mean or another max instead of max then sum, and using the hyperparemeters described in the ColBERT paper such as lower learning rate and gradient clipping. These experiments didn't perform as well as the main ones listed here.

## Results
|                     | Train loss | Val MRR | Test MRR |
|---------------------|------------|---------|----------|
| Original            | 0.7406     | 0.5628  | 0.4883   |
| Maxsim over docs    | 0.8066     | 0.508   | 0.4339   |
| Maxsim over lengths | 0.983      | 0.526   | 0.4593   |
| Maxsim over both    | 0.6133     | **      | 0.3712   |

*Everything trained with batch size 32, 10 epochs, and full training dataset \
**Couldn't calculate val MRR on the full set since the pairwise einsum between all queries and targets would have taken 64 hours to compute

WandB link: https://wandb.ai/xyznlp/lightning_logs/reports/Replication-Project--Vmlldzo0MDI5MjEw
### Observations
* All maxsims are worse than the original in terms of MRR scores
  * However, "maxsim over both" has a lower loss
* "Maxsim over lengths" achieves better MRR than "maxsim over docs" despite a higher loss
* "Maxsim over both" scores the worst test MRR despite having the lowest train loss
* Loss does not appear to be 1-1 correlated with eval MRR
* Differences between train and eval setting:
  * Train always has 2 samples per author in each batch, evaluation uses separate query and target dataloaders with only 1 sample per author in each batch
  * Train randomly samples 1 to 16 docs, evaluation always uses 16 docs
  * Train examples contain more padding on average
  * Eval computes a distance matrix for the entire validation/test set before ranking (over 100k authors), training computes the distance matrix for a single batch when calculating loss (only has batch_size authors)