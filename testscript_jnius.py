from scoring_jnius import ProBoundModel
import numpy as np
import timeit

bases = list("ACGT")
seqs = ["".join(np.random.choice(bases, size=1000)) for i in range(1000)]

model = ProBoundModel("test_input/fit.sox2.json", fitjson=True)
model.remove_binding_mode(0)

import cProfile

with cProfile.Profile() as pr:
    time_new = timeit.timeit(
        lambda: model.score_binding_mode_scores(seqs,
                                                score_format="profile",
                                                profile_aggregate="max"
                                                ),
        number=1)

    pr.print_stats(sort="tottime")

print(f"Time of implementation: {time_new}")