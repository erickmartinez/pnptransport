import numpy as np


def evaluate_arrhenius(a0: float, Ea: float, temp: float) -> float:
    """
    Evaluate an Arrhenius variable

    Parameters
    ----------
    a0 : float
        The exponential prefactor
    Ea : float
        An activation energy in eV
    temp : float
        The temperature

    Returns
    -------
    x : float
        The evaluated variable
    """
    kT = temp * (1.38064852 / 1.60217662) * 1E-4  # The thermal energy in eV
    return a0 * np.exp(-Ea / kT)


def format_time_str(time_s: float):
    """
    Returns a formatted time string

    Parameters
    ----------
    time_s : float
        The time in seconds


    Returns
    -------
    timeStr : str
        A string representing the time
    """
    time_s = abs(time_s)
    min2s = 60
    hr2s = min2s * 60
    day2s = hr2s * 24
    mon2s = day2s * 30.42
    yr2s = day2s * 365

    years = np.floor(time_s / yr2s)
    time_s = time_s - years * yr2s
    months = np.floor(time_s / mon2s)
    time_s = time_s - months * mon2s
    days = np.floor(time_s / day2s)
    time_s = time_s - days * day2s
    hrs = np.floor(time_s / hr2s)
    time_s = time_s - hrs * hr2s
    mins = np.floor(time_s / min2s)
    time_s = time_s - mins * min2s

    #    time_str = "%01d Y %02d M %02d d %02d:%02d:%02d" % (years, \
    #        months,days,hrs,mins,time_s)
    if years >= 1:
        time_str = "%01dY %02dM %02dd %02d:%02d:%02d" % (years,
                                                         months, days, hrs, mins, time_s)
    elif months >= 1:
        time_str = "%02dM %02dd %02d:%02d:%02d" % (months, days, hrs, mins, time_s)
    elif days >= 1:
        time_str = "%02dd %02d:%02d:%02d" % (days, hrs, mins, time_s)
    elif hrs >= 1:
        time_str = "%02d:%02d:%02d" % (hrs, mins, time_s)
    elif mins >= 1:
        time_str = "%02d:%02d" % (mins, time_s)
    else:
        time_str = "%02d" % time_s
    return time_str


def fit_arrhenius(temperature_axis, y, **kwargs):
    """
    Fits the experimental data to an Arrhenius relationship

    Parameters
    ----------
    temperature_axis: [double]
        The temperature axis
    y: [double]
        The dependent variable
    **kwargs
        inverse_temp: boolean
            True if the units of the temperature are 1/T
        temp_units: string
            The units of the temperature. Valid units are K and °C
    """

    temp_factor = kwargs.get('temp_factor', 1)
    std = kwargs.get('std', None)
    p0 = kwargs.get('p0', None)
    fscale = kwargs.get('fscale', 0.1)
    kB = 8.6173303E-5  # eV/K

    import pnptransport.confidence as cf
    import scipy.optimize as opt
    from scipy.linalg import svd

    x = temperature_axis.copy()

    if std is not None:
        sigma = 1 / std
    else:
        sigma = np.ones_like(y)

    def func(X, p):
        a = p[0]
        b = p[1]
        return X * a + b

    def fobj(p):
        return sigma * (func(x, p) - y) * sigma

    if p0 is None:
        p0 = [-10, 0.5]

    all_tol = np.finfo(np.float64).eps
    res: opt.OptimizeResult = opt.least_squares(
        fobj, p0, xtol=all_tol, ftol=all_tol, gtol=all_tol,
        method='trf',
        jac='3-point',
        # loss='cauchy',
        tr_options={'regularize': True},
        x_scale='jac',
        f_scale=fscale,
        max_nfev=20000
    )

    xfit = np.linspace(min(x), max(x))
    #    yfit = func(xfit,*popt)

    # Get the confidence interval for the fitted parameters
    popt = res.x

    ysize = len(res.fun)
    cost = 2 * res.cost  # res.cost is half sum of squares!
    s_sq = cost / (ysize - popt.size)

    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, VT = svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s ** 2, VT)
    pcov = pcov * s_sq

    ci = cf.confint(len(x), popt, pcov)
    # Get the prediction bands for the solution
    yfit, lpb, upb = cf.predint(xfit, x, y, func, res, mode='observation')

    A0 = np.exp(popt[1])
    Ea = -popt[0] * kB * temp_factor

    A0_err = np.abs(np.exp(ci[1, 1]) - np.exp(popt[1]))
    a_err = np.amax(np.abs(ci[0, :] - popt[0]))

    Ea_err = a_err * kB * temp_factor

    return {'A0': A0,
            'A0_err': A0_err,
            'Ea': Ea,
            'Ea_err': Ea_err,
            'popt': popt,
            'pcov': pcov,
            'ci': ci,
            'xfit': xfit,
            'yfit': yfit,
            'lpb': lpb,
            'upb': upb}


def latex_format(x, digits=2) -> str:
    """
    Creates a latex string for matplotlib plots.

    Parameters
    ----------
    x: str
        The value to be formatted
    digits: int
        The number of digits to round up to.

    Returns
    -------
    str:
        The math-ready string
    """
    fmt_dgts = '%%.%df' % digits
    fmt_in = '%%.%dE' % digits
    x_str = fmt_in % x
    x_sci = (np.array(x_str.split('E'))).astype(np.float)
    if digits == 0:
        return r'$\mathregular{10^{%d}}$' % x_sci[1]
    else:
        ltx_str = fmt_dgts % x_sci[0]
        ltx_str += r'$\mathregular{\times 10^{%d}}$' % x_sci[1]
        return ltx_str


def latex_format_with_error(num, err):
    num_str = '%.9E' % num
    num_sci = (np.array(num_str.split('E'))).astype(np.float)
    if np.isinf(err):
        return r'%.2f$\mathregular{\times 10^{%d}}$ $\pm$ INF' % (num_sci[0], num_sci[1])
    err_str = '%.9E' % err
    err_sci = (np.array(err_str.split('E'))).astype(np.float)

    if err_sci[1] == num_sci[1]:
        digits_num = 1  # 0
        digits_err = 0
    elif err_sci[1] < num_sci[1]:
        digits_num = num_sci[1] - err_sci[1]  # - 1
        digits_err = digits_num  # + 1
    else:
        digits_num = num_sci[1] - err_sci[1]  # - 1
        digits_err = digits_num  # + 1

    if np.abs(num_sci[1]) <= 2:
        err_sci[0] = err_sci[0] / np.power(10, digits_err)
        if digits_num < 0 or digits_err < 0:
            digits_num = 2  # 1
            digits_err = 1  # 0

        if num_sci[1] < 0:
            digits_num = abs(num_sci[1])
        if err_sci[1] < 0:
            digits_err = abs(err_sci[1])

        fmt_num = '%%.%df' % digits_num
        fmt_err = '%%.%df' % digits_err
        factor = np.power(10, num_sci[1])
        ltx_str = fmt_num % (num_sci[0] * factor)
        ltx_str += r'$\pm$'
        ltx_str += fmt_err % (err_sci[0] * factor)
    else:
        if digits_num < 0 or digits_err < 0:
            digits_num = 0
            digits_err = 0
        if np.abs(num_sci[1] - err_sci[1]) < 5:
            err_sci[0] = err_sci[0] / np.power(10, (num_sci[1] - err_sci[1]))
            fmt_num = '%%.%df' % digits_num
            fmt_err = '%%.%df' % digits_err
            ltx_str = r'(' + fmt_num % num_sci[0]
            ltx_str += r'$\pm$'
            ltx_str += fmt_err % err_sci[0] + ')'
            ltx_str += r'$\mathregular{\times 10^{%d}}$' % num_sci[1]
        else:
            ltx_str = '%.2f' % num_sci[0]
            ltx_str += r'$\mathregular{\times 10^{%d}}$' % num_sci[1]
            ltx_str += r'$ \pm $'
            ltx_str += '%.2f' % err_sci[0]
            ltx_str += r'$\mathregular{\times 10^{%d}}$' % err_sci[1]

    return ltx_str


def get_indices_at_values(x: np.array, requested_values: np.array) -> np.ndarray:
    """
    Constructs an array of valid indices in the x array corresponding to the requested values

    Parameters
    ----------
    x: np.array
        The array from which the indices will be drawn
    requested_values: np.array

    Returns
    -------
    np.array
        An array with the indices corresponding to the requested values
    """
    result = np.empty(len(requested_values), dtype=int)
    for i, v in enumerate(requested_values):
        result[i] = int((np.abs(v - x)).argmin())
    return result


def tau_c(D: float, E: float, L: float, T: float) -> float:
    """
    Estimates the characteristic constant for the Nernst-Planck
    equation in the low concentration approximation

        .. math:: \\tau_c = \\frac{2D}{\\mu^2 E^2} + \\frac{L}{\\mu E} \\left[1 ± 2\\left( \\frac{kT}{q E L}\\right)^{1/2}\\right]

    Since  :math:`\\mu = qD/kT`

        .. math:: \\tau_c = \\left( \\frac{2}{D} \\right) X^2 + \\left( \\frac{l}{D} \\right) X  \\left[ 1 ± 2 \\left( \\frac{X}{L} \\right)^{1/2} \\right],

    with :math:`X = kT/qE`

    Parameters
    ----------
    D: float
        The diffusion coefficient in cm^2/s
    E: float
        The electric field in MV/cm = 1E6 V/cm
    L: float
        The distance in cm
    T: float
        The temperature in °C

    Returns
    -------
    float:
        The characteristic time in s
    """
    TK = T + 273.15
    kB_red = 8.6173303  # 1E-5 eV/K

    kTq_red = kB_red * TK  # x 1E-5 V

    x = kTq_red / E  # x 1E-5 V x 1E-6 cm/V = 1E-11 cm

    tau1 = 1.0E-11 * x * (L / D) + (2.0E-22 * np.power(x, 2.0) / D) * (1.0 - np.sqrt(1.0 + 1E-11 * (L / x)))
    tau2 = 1.0E-11 * x * (L / D) + (2.0E-22 * np.power(x, 2.0) / D) * (1.0 + np.sqrt(1.0 + 1E-11 * (L / x)))

    if tau1 <= 0:
        return tau2
    elif tau2 <= 0:
        return tau1
    else:
        return min(tau1, tau2)