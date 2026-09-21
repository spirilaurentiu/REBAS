# Imports
import numpy as np
import scipy.stats as stats

# Parse arguments
import argparse

parser = argparse.ArgumentParser(description="Generate a normal distributed random sample")
parser.add_argument("--mean", type=float, default=0, help="Mean of the distribution")
parser.add_argument("--std", type=float, default=1, help="Standard deviation of the distribution")
parser.add_argument("--sampleSize", type=int, default=100, help="Number of samples")
parser.add_argument("--scaleFactor", type=float, default=2, help="Scaling factor for the difference from the mean")
args = parser.parse_args()

# Main function
def main():

    scalefactor = args.scaleFactor

    # Generate a normal distributem random sample
    X = np.random.normal(loc=args.mean, scale=args.std, size=args.sampleSize)

    # Scale the difference from the average by a scaling factor s
    X_avg = np.mean(X)
    Y = X_avg + scalefactor * (X - X_avg)

    # Get the cumulative running mean of X
    cumulative_mean_X = np.array([np.mean(X[:i]) for i in range(1, len(X) + 1)])

    # Get the cumulative running variance of X
    cumulative_var_X = np.array([np.var(X[:i]) for i in range(1, len(X) + 1)])

    # Get the cumulative running mean of Y
    cumulative_mean_Y = np.array([np.mean(Y[:i]) for i in range(1, len(Y) + 1)])

    # Get the cumulative running variance of Y
    cumulative_var_Y = np.array([np.var(Y[:i]) for i in range(1, len(Y) + 1)])

    # Get the theoretical variance of X as a sample distribution (Var(X) = std^2 / n)
    theo_var_of_mean_X = np.array([args.std ** 2 / i for i in range(1, len(X) + 1)])


    # Get covariance of X and cumulative mean of X
    covariance_X_mean_X = np.array([np.cov(X[:i], cumulative_mean_X[:i])[0, 1] for i in range(1, len(X) + 1)])

    #Print X and Y next to each other
    print("X\tTheoretical_Variance_Of_Mean_X\tCumulative_Mean_X\tCovariance_X_Mean_X\tCumulative_Variance_X\tY\tCumulative_Mean_Y\tCumulative_Variance_Y")
    for xi, x in enumerate(X):
        x_mean_var_theo = theo_var_of_mean_X[xi]
        x_mean = cumulative_mean_X[xi]
        x_cov_mean = covariance_X_mean_X[xi]
        x_var = cumulative_var_X[xi]
        y = Y[xi]
        y_mean = cumulative_mean_Y[xi]
        y_var = cumulative_var_Y[xi]
        print(f"{x:.4f}\t{x_mean_var_theo:.4f}\t{x_mean:.4f}\t{x_cov_mean:.4f}\t{x_var:.4f}\t{y:.4f}\t{y_mean:.4f}\t{y_var:.4f}")

    invN = 1.0 / args.sampleSize

    meanVarFactor = 1.0 - (2.0*scalefactor) + (scalefactor**2)
    varFactor = meanVarFactor * invN
    bias = varFactor * (args.std ** 2)
    myTotalVarFactor = (scalefactor**2 + bias)

    Y_var_theo = myTotalVarFactor * (args.std ** 2)
    print(f"Theoretical Variance of Y: {Y_var_theo:.4f}")

    ai_var_factor = invN + (scalefactor**2) + (invN * (scalefactor**2))
    ai_var = ai_var_factor * (args.std ** 2)
    print(f"AI Variance of Y: {ai_var:.4f}")

if __name__ == "__main__":
    main()