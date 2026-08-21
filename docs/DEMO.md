# Current demo

`results/demos/piu_original_fresh_seed1399_v1/piu_information_acquisition.mp4`
is the only retained demo.

It shows one disposable original-cluttered-scene trace:

```text
Place the butter in the basket
  -> OPEN_CONTAINER
  -> fresh frozen pi05_libero option
  -> singleton public-RGB REVEALED
  -> belief update
  -> MOVE_CLOSER
```

It demonstrates a real information-acquisition chain. It does not execute the
post-replan option, does not place the butter, and is not clean or sealed
evidence. Agentview is a stock policy input; evaluator-only values appear only
in the trace section explicitly marked after controller termination.
