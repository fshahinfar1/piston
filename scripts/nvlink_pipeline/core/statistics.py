from typing import *


def stats(lst: List[float]) -> Tuple[float, float, float, float, float]:
    if not lst:
        return "No data"
    S = sorted(lst)
    mean = sum(S) / len(S)
    std = (sum((x - mean) ** 2 for x in S) / len(S)) ** 0.5
    mid = S[len(S) // 2] if len(S) % 2 == 1 else (S[len(S) // 2 - 1] + S[len(S) // 2]) / 2
    return mean, std, mid, S[-1], len(S)


class ExecutionStatistics:
    def __init__(self, num_stages: int):
        self.stage_exec_times: Dict[int, List[int]] = {
            i: [] for i in range(num_stages)
        }
        self.hidden_state_transfer_times: Dict[int, List[int]] = {
            i: [] for i in range(num_stages)
        }

    def report(self) -> None:
        num_stages = len(self.stage_exec_times)
        print("Execution Statistics:")
        for stage_index, exec_times in self.stage_exec_times.items():
            # mid and std and avg of list
            mean, std, mid, _max, count = stats(exec_times)
            print(f"Stage {stage_index} execution times:")
            print(f"\t\tmean={mean:.4f}, std={std:.4f}, mid={mid:.4f}, max={_max:.4f}, count={count}")
        for stage_index, transfer_times in self.hidden_state_transfer_times.items():
            mean, std, mid, _max, count = stats(transfer_times)
            print(f"hidden state transfer from stage {(stage_index - 1) % num_stages} to stage {stage_index}:")
            print(f"\t\tmean={mean:.4f}, std={std:.4f}, mid={mid:.4f}, max={_max:.4f}, count={count}")
