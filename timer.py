# timer.py
import time
import pandas as pd
from tqdm.auto import tqdm

total_steps = 1 # change this to the number of total steps in your notebook
pbar = tqdm(total=total_steps, desc="Notebook pipeline", unit="step")
timings = []


def start_timer():
    return time.perf_counter()


def end_timer(step_name, t0):
    dt = time.perf_counter() - t0
    timings.append({"step": step_name, "seconds": dt})
    pbar.update(1)
    pbar.set_postfix(last=f"{dt:.2f}s")
    print(f"{step_name}: {dt:.2f}s")

    """ 
    USAGE INSTRUCTIONS:
    change the total_steps variable to the number of total steps in your notebook
    start each cell with the t0 line, end each cell with the end_timer line (change the name in quotes).
    Put the last bit of code at the end of your notebook.
    """


t0 = start_timer() # start each cell with this line
end_timer("Reading data", t0) # end each cell with this line, changing the step name as needed

# put at the end of your notebook to see the timings for each step and the total time
pbar.close()
df_times = pd.DataFrame(timings)
display(df_times)
print(f"Total time: {df_times['seconds'].sum():.2f}s")
