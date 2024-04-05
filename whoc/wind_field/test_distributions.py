import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # generate probabilities for known inputs from univariate gaussian pdf
    mu = 8
    sigma = 4.0
    
    lb_angle = 60 * np.pi / 180.0
    ub_angle = 120 * np.pi / 180.0
    # lb_angle = -45 * np.pi / 180.0
    # ub_angle = 45 * np.pi / 180.0

    lb_tan = min(np.tan(lb_angle), np.tan(ub_angle))
    ub_tan = max(np.tan(lb_angle), np.tan(ub_angle))

    err = []
    sample_sizes = np.linspace(100, 10000, 50)
    for n_samples in sample_sizes:
        ratio_range = np.linspace(-20, 20, int(n_samples))
        dratio = ratio_range[1] - ratio_range[0]

        # pdf_ratio = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
        # probs_ratio = pdf_ratio(ratio_range)
        probs_ratio = norm.pdf(ratio_range, loc=mu, scale=sigma)
        
        # compute nonlinear function of inputs
        angle_range = np.arctan(ratio_range)
        probs_angle = probs_ratio[np.argsort(angle_range)]
        angle_range = np.sort(angle_range)
        dangle = np.diff(angle_range)

        # compute cdf for angular bounds and for tranfsormed bounds
        angle_shift = -(np.pi/2) - lb_angle
        lb_angle += angle_shift
        ub_angle += angle_shift
        # angle_range += angle_shift
        dangle = np.diff(angle_range)


        lb_angle_ratio = np.argmin(abs(angle_range - lb_angle))
        ub_angle_ratio = np.argmin(abs(angle_range - ub_angle))
        cdf_angle = sum(probs_angle[lb_angle_ratio:ub_angle_ratio+1] * dangle[lb_angle_ratio:ub_angle_ratio+1])
        
        lb_tan_ratio = np.argmin(abs(ratio_range - lb_tan))
        ub_tan_ratio = np.argmin(abs(ratio_range - ub_tan))
        cdf_tan = sum(probs_ratio[lb_tan_ratio:ub_tan_ratio+1] * dratio)
        # cdf_tan_true = norm.cdf(ub_tan, loc=mu, scale=sigma) - norm.cdf(lb_tan, loc=mu, scale=sigma)
        err.append(abs(cdf_angle - cdf_tan))


    plt.plot(sample_sizes, err)
    plt.show()
    # print(cdf_angle)
    # print(cdf_tan)
    # print(cdf_tan_true)

