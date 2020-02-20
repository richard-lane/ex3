"""
Generate random numbers with particular distributions

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

import timeit


def analytical_sin_random_numbers(num_points):
    """
    Generate an array of numbers between 0 and pi following a sin(x) distribution

    """
    uniform_random_numbers = [np.random.random() for i in range(num_points)]
    return np.array([np.arccos(1 - 2 * x) for x in uniform_random_numbers])


def accept_reject_random_numbers(num_points, func, xlim, ylim):
    """
    Generate random numbers from the distribution described by func(x) using the accept-reject method.

    xlim and ylim should be iterables describing the domain/range of the function to be considered.

    """
    # Check that our x and y limits are iterables of the right size
    if len(xlim) != 2 or len(ylim) != 2:
        raise Exception("Must provide x and y ranges of 2 values for accept-reject")

    # Find where our x and y range start and how long they are; will use these for generating our points
    min_x = min(xlim)
    min_y = min(ylim)
    domain_size = max(xlim) - min_x
    range_size = max(ylim) - min_y

    generated_numbers = np.zeros(num_points)
    num_accepted_points = 0

    while num_accepted_points < num_points:
        # Generate a random point in the defined (x, y) box
        # This could be optimised to use a more intelligent shape but this will do for now
        generated_x = min_x + domain_size * np.random.random()
        generated_y = min_y + range_size * np.random.random()

        # Find the actual value of our function at our x point
        func_val = func(generated_x)

        # If our y boundaries do not cover the whole range of possible func_vals, our accept-reject will not work and so
        # we throw an exception
        if func_val < min_y or func_val > max(ylim):
            # A better implementation would define custom exception types for this sort of thing
            raise Exception(
                f"Generated value {func_val} outside of provided range {ylim}. Are your limits correct?"
            )

        # Check if our generated point is accepted
        if generated_y < func_val:
            generated_numbers[num_accepted_points] = generated_x
            num_accepted_points += 1

    return generated_numbers


def plot_hist(data, title, n_bins=100):
    """
    Plot a histogram from the provided data

    """
    plt.hist(data, bins=n_bins)
    plt.title(title)
    plt.show()


def q1a():
    """
    Generate and plot angles distributed according to sin(x) using an analytical method and accept-reject

    """
    num_points = 100000

    analytical_angles = analytical_sin_random_numbers(num_points)
    acc_rej_angles = accept_reject_random_numbers(
        num_points, np.sin, (0, np.pi), (0, 1)
    )

    plot_hist(
        analytical_angles,
        r"Angles $\theta$ generated proportional to sin($\theta$) analytically",
    )
    plot_hist(
        acc_rej_angles,
        r"Angles $\theta$ generated proportional to sin($\theta$) using accept-reject",
    )


def time_methods(max_exponent):
    """
    Time evalution of 10^n points with accept-reject and analytical analysis for n up to max_exponent
    Then make a plot of the times

    """
    acc_rej_times = []
    analytical_times = []
    n = []

    for num_points in np.logspace(1, max_exponent, num=max_exponent):
        # Using a lambda here adds a bit of overhead but it shouldn't be too much of an issue as the functions aren't
        # that fast to evaluate
        # Note also that a better implementation would do more measurements with lower numbers of points (the timing
        # uncertainty will be larger), but this gives a decent idea of how the timing scales for each method.
        print(f"Running acc-rej with \t\t{num_points} points")
        acc_rej_time = min(
            timeit.Timer(
                lambda: accept_reject_random_numbers(
                    int(num_points), np.sin, (0, np.pi), (0, 1)
                )
            ).repeat(number=3, repeat=3)
        )

        print(f"Running analytical method with \t{num_points} points\n")
        analytical_time = min(
            timeit.Timer(lambda: analytical_sin_random_numbers(int(num_points))).repeat(
                number=3, repeat=3
            )
        )

        acc_rej_times.append(acc_rej_time)
        analytical_times.append(analytical_time)
        n.append(num_points)

    # Plot both on a log scale and hopefully something sensible happens
    plt.plot(n, acc_rej_times, "k.", label="Accept-reject")
    plt.plot(n, analytical_times, "r.", label="Analytical")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of points generated")
    plt.ylabel("Time/s")
    plt.legend()
    plt.title("Time to generate sets of random numbers")
    plt.show()


def fit_methods(num_points):
    """
    Generate sin(x) data using accept-reject and an analytical method, then overlay a plot of the expected distribution

    Should also do a fit to a sin/chi squared/some test of how good our fit is but i cba

    """
    # Number of histogram bins to use
    num_bins = 100
    bin_limits = [i for i in np.linspace(0, np.pi, num=num_bins)]
    bin_centres = [0.5*(bin_limits[i] + bin_limits[i+1]) for i in range(num_bins-1)]

    analytical_angles = analytical_sin_random_numbers(num_points)
    acc_rej_angles = accept_reject_random_numbers(
        num_points, np.sin, (0, np.pi), (0, 1)
    )

    # Create numpy histograms of our data
    analytical_hist, analytical_bins = np.histogram(analytical_angles, bin_limits)
    acc_rej_hist, acc_rej_bins = np.histogram(acc_rej_angles, bin_limits)
    # Our binning doesn't necessarily have to be the same but it would be nice
    assert np.allclose(analytical_bins, acc_rej_bins)

    # Our analytical_hist and acc_rej_hist are lists containing the numbers of points in each bin
    # Compare this with the expected distribution now...

    # Expect our distributions to look like sin(x), times a normalisation
    # Parametrise this as a * sin(b*x)
    theta = [i for i in np.linspace(0, np.pi, num=num_bins)]
    expected_dist = lambda x, a, b: a * np.sin(b * x)  # named lambda lol
    expected_a = num_points / num_bins * np.pi / 2
    expected_b = 1

    analytical_popt, analytical_pcov = optimize.curve_fit(expected_dist, bin_centres, analytical_hist)
    acc_rej_popt, acc_rej_pcov = optimize.curve_fit(expected_dist, bin_centres, acc_rej_hist)

    # Do some output
    print(f"Analytical:\n\ta =\t{analytical_popt[0]}+-{analytical_pcov[0][0]}")
    print(f"\tb =\t{analytical_popt[1]}+-{analytical_pcov[1][1]}")

    print(f"\nAcc-Rej:\n\ta =\t{acc_rej_popt[0]}+-{acc_rej_pcov[0][0]}")
    print(f"\tb =\t{acc_rej_popt[1]}+-{acc_rej_pcov[1][1]}")

    print(f"\nExpected values\n\t{expected_a}\n\t{expected_b}")




def q1b():
    """
    Do some stuff

    """
    # Perform a fit to each and see how it compares to the analytical solution

    # The argument here refers to the maximum number of points to investigate. e.g. "5" would mean generate up to
    # 100,000 points
    # time_methods(5)

    fit_methods(100000)


def main():
    try:
        q1a()
        q1b()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
