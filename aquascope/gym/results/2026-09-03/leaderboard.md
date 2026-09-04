## HydroGym leaderboard, 2026-09-03 (60 tasks, seed 2026)

60 tasks, 280 runs, 5 agent-model pairs. Accuracy is the share of solvable tasks on which the agent picked the expected playbook branch (a keyless decline counts as wrong); declined is the share of unsolvable tasks the agent refused; false declines are solvable tasks it refused; gates and tools are the fractions of the key's gates evaluated and tools called (means over solvable tasks); accuracy on test is over the solvable tasks of the held-out split (its size in brackets); an error or a timeout counts as wrong.

| agent | model | tasks (solvable + unsolvable) | accuracy | accuracy on test | declined unsolvable | false declines | gates | tools | tokens/task | s/task | cost USD | errors | timeouts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ask | claude-sonnet-5 | 40 (25 + 15) | 68 % | 75 % (8) | 100 % | 16 % | 0 % | 33 % | 30,810 | 61.9 | 3.017 | 1 | 1 |
| team | keyless | 60 (45 + 15) | 100 % | 100 % (18) | 100 % | 0 % | 100 % | 99 % | 0 | 15.5 | 0.000 | 0 | 0 |
| team | claude-haiku-4-5 | 60 (45 + 15) | 98 % | 100 % (18) | 100 % | 0 % | 100 % | 99 % | 4,966 | 23.1 | 0.406 | 1 | 1 |
| team | claude-sonnet-5 | 60 (45 + 15) | 100 % | 100 % (18) | 100 % | 0 % | 100 % | 99 % | 6,575 | 25.0 | 1.145 | 0 | 0 |
| tree | none | 60 (45 + 15) | 100 % | 100 % (18) | 100 % | 0 % | 100 % | 100 % | 0 | 0.0 | 0.000 | 0 | 0 |

Correct on solvable tasks by expected branch (correct / n):

| agent | model | at_gauge | at_site | regional | short_record | well |
|---|---|---|---|---|---|---|
| ask | claude-sonnet-5 | - | 10 / 11 | 1 / 6 | 0 / 2 | 6 / 6 |
| team | keyless | 15 / 15 | 11 / 11 | 10 / 10 | 2 / 2 | 7 / 7 |
| team | claude-haiku-4-5 | 15 / 15 | 11 / 11 | 10 / 10 | 1 / 2 | 7 / 7 |
| team | claude-sonnet-5 | 15 / 15 | 11 / 11 | 10 / 10 | 2 / 2 | 7 / 7 |
| tree | none | 15 / 15 | 11 / 11 | 10 / 10 | 2 / 2 | 7 / 7 |

Cost is estimated from the tokens the provider reported and a small table of list prices (aquascope.gym.bench.PRICES_USD_PER_MTOK, mid-2026); prices change, cache and batch discounts are not modelled, and a model not in the table gets no estimate.
