"""Public CPU RNG/distribution observations; no runtime implementation source is inspected."""
import warnings
warnings.simplefilter("ignore")
import torch
torch.set_num_threads(1)
for size in [3, 16, 51866]:
    torch.manual_seed(0)
    uniform = torch.rand(size)
    torch.manual_seed(0)
    exponential = torch.empty(size).exponential_()
    exp_state = torch.get_rng_state()
    torch.manual_seed(0)
    selected = torch.multinomial(torch.ones(size), 1).item()
    multinomial_state = torch.get_rng_state()
    consumed = []
    for count in [1, 2, 3, 4, 8, 16, size, size * 2]:
        torch.manual_seed(0)
        torch.rand(count)
        if torch.equal(torch.get_rng_state(), multinomial_state):
            consumed.append(count)
    print({"size": size, "uniform_first": uniform[:5].tolist(), "exponential_first": exponential[:5].tolist(), "multinomial": selected, "exp_race_index": int(exponential.argmin()), "same_rng_state": torch.equal(exp_state, multinomial_state), "equivalent_rand_counts": consumed})
