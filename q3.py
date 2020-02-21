"""
hello
"""
import numpy as np


def find_background_mean():
    """
    We know the background is 4.8+-0.5; draw a number from this distribution to find the desired background mean to use
    in each experiment

    """
    return np.random.normal(4.8, 0.5)


def perform_experiment(signal_cross_section):
    """
    Perform a simulated counting experiment with the given cross section, return the number of events

    """
    luminosity = 10
    # Find this experiment's signal and background counts
    background_counts = np.random.poisson(find_background_mean())
    signal_counts = np.random.poisson(luminosity * signal_cross_section)

    return signal_counts + background_counts


def main():
    counts = []
    errs = []
    for signal_cross_section in np.linspace(0, 3):
        tmp = []
        for i in range(10000):
            tmp.append(perform_experiment(signal_cross_section))
        counts.append(np.mean(tmp))
        errs.append(np.std(tmp))

    for i in range(len(counts)):
        print(f"{counts[i]}\t+-\t{errs[i]}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
