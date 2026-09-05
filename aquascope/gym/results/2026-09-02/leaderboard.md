## HydroGym smoke, 2026-09-02

12 tasks, 28 runs, 3 agent-model pairs. Accuracy is the share of solvable tasks on which the agent picked the expected playbook branch (a keyless decline counts as wrong); declined is the share of unsolvable tasks the agent refused; false declines are solvable tasks it refused; gates and tools are the fractions of the key's gates evaluated and tools called (means over solvable tasks).

| agent | model | tasks (solvable + unsolvable) | accuracy | accuracy on test | declined unsolvable | false declines | gates | tools | tokens/task | s/task | cost USD | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ask | claude-sonnet-5 | 8 (6 + 2) | 67 % | - (0) | 100 % | 0 % | 0 % | 31 % | 62,314 | 58.4 | 1.113 | 0 |
| team | claude-sonnet-5 | 8 (6 + 2) | 100 % | - (0) | 100 % | 0 % | 100 % | 100 % | 6,437 | 25.9 | 0.154 | 0 |
| tree | none | 12 (9 + 3) | 100 % | - (0) | 100 % | 0 % | 100 % | 100 % | 0 | 0.0 | 0.000 | 0 |

Correct on solvable tasks by expected branch (correct / n):

| agent | model | at_gauge | at_site | regional | short_record | well |
|---|---|---|---|---|---|---|
| ask | claude-sonnet-5 | 2 / 2 | 1 / 1 | 0 / 1 | 0 / 1 | 1 / 1 |
| team | claude-sonnet-5 | 2 / 2 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| tree | none | 3 / 3 | 2 / 2 | 2 / 2 | 1 / 1 | 1 / 1 |

Cost is estimated from the tokens the provider reported and a small table of list prices (aquascope.gym.bench.PRICES_USD_PER_MTOK, mid-2026); prices change, cache and batch discounts are not modelled, and a model not in the table gets no estimate.
