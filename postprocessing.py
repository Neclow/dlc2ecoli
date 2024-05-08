# pylint: disable=invalid-name, line-too-long, unsubscriptable-object
"""
Preprocessing functions: data filtering, smoothing and aligning.

Some functions are inspired by DLC2Kinematics (https://github.com/AdaptiveMotorControlLab/DLC2Kinematics)
"""
import numpy as np
import pandas as pd

from scipy import signal

from config import FPS, MIN_LIKELIHOOD
from geom_utils import is_point_inside_convex_quadrilateral, scale_rectangle, shoelace


def clean_data(df, dropna=False, limit_direction="forward", limit=FPS // 2):
    """Clean data:
     - separate (x, y) from likelihood data
     - remove low-likelihood predictions
     - apply some interpolation for missing values

    Parameters
    ----------
    df : pandas.DataFrame
        Raw data

    Returns
    -------
    pos_df : pandas.DataFrame
        Bodypart positions at each frame from DLC
    lik_df : pandas.DataFrame
        Likelihood of DLC predictions for each bodypart and frame
    """
    # if smoothed:
    #     df = df.apply(smooth_signal, **kwargs)

    pos_df, lik_df = [
        x
        for _, x in df.groupby(
            df.columns.get_level_values("coords").str.contains("likelihood"), axis=1
        )
    ]

    # Replace low-likelihood data using linear interpolation
    low_lik_mask = pos_df.apply(
        lambda x: x.mask(
            lik_df.iloc[:, pos_df.columns.get_loc(x.name) // 2] < MIN_LIKELIHOOD
        )
    )

    if dropna:
        low_lik_mask.dropna(inplace=True)

    return (
        low_lik_mask.interpolate(limit_direction=limit_direction, limit=limit),
        lik_df,
    )


def smooth_signal(x, mode="savgol", window_len=15, polyorder=7, **kwargs):
    """Smooth a signal with different options.

    Available smoothing techniques:
    * Savitzky-Golay filter
    * Median filter
    * Boxcar/Bartlett-windowed moving averages

    Might add Hampel filtering in a future version

    Sources:
    # https://www.jneurosci.org/content/jneuro/early/2022/02/22/JNEUROSCI.0938-21.2022.full.pdf

    Parameters
    ----------
    x : array-like
        Input signal
    mode : str, optional
        Smoothing mode, by default 'savgol' (Savitzky-Golay filter)
    window_len : int, optional
        Window length, by default 7
    polyorder : int, optional
        The order of the polynomial used to fit the samples. polyorder must be less than window_length.
    **kwargs : kwargs
        Other arguments passed to the smoothing function

    Returns
    -------
    numpy.ndarray
        Smoothed signal
    """

    if mode == "boxcar":
        w = signal.boxcar(M=window_len)
        return np.convolve(x, w / w.sum(), mode="same", **kwargs)
    if mode == "bartlett":
        w = signal.bartlett(M=window_len)
        return np.convolve(x, w / w.sum(), mode="same", **kwargs)
    if mode == "medfilt":
        return signal.medfilt(x, kernel_size=(window_len,), **kwargs)
    if mode == "savgol":
        return signal.savgol_filter(
            x,
            window_length=window_len,
            polyorder=kwargs.pop("polyorder", polyorder),
            **kwargs,
        )

    raise ValueError(f"Unknown smoothing mode: {mode}.")


def derive(x, order=1, **kwargs):
    """Compute the n-th derivative of a signal using a Savitzky-Golay filter

    If order = 0, simply applies smoothing

    Parameters
    ----------
    x : array-like
        Input array
    order : int, optional
        Derivative order, by default 1
    **kwargs : kwargs
        All other arguments are passed to smooth_signal

    Returns
    -------
    array-like
        n-th derivative of x
    """
    return smooth_signal(x, mode="savgol", deriv=order, delta=FPS, **kwargs)


def filter_outliers(df, max_zscore=3):
    """Replace z-score outliers by NaN.

    True if abs(z-score) > max_zscore, False otherwise

    Parameters
    ----------
    df : pandas.DataFrame
        Input 2D data, columns = features

    Returns
    -------
    pandas.DataFrame
        Filtered data, where outliers are replaced by NaNs
    """
    df_scaled = df.sub(df.mean()).div(df.std())

    outlier_mask = df_scaled.applymap(lambda x: abs(x) > max_zscore)

    return df.mask(outlier_mask)


def get_length(pos_df, a="head", b="saddle"):
    # Calculate the distance between two body parts
    a_and_b = pos_df.loc[:, pos_df.columns.get_level_values("bodyparts").isin((a, b))]

    # x = -x, y = -y for b coordinate
    a_and_b.iloc[:, 2::4] = -a_and_b.iloc[:, 2::4]
    a_and_b.iloc[:, 3::4] = -a_and_b.iloc[:, 3::4]

    a_to_b = (
        a_and_b.groupby(level=[0, 2], axis=1)
        .sum(min_count=2)  # head_x - saddle_x, head_y - saddle_y
        .pow(2)  # dx^2, dy^2
        .groupby(level=0, axis=1)
        .sum(min_count=2)  # dx^2 + dy^2
        .apply(np.sqrt)  # qrt
    )

    return a_to_b


def get_travel(pos_df):
    # Calculate center position
    pos_centre = (
        pos_df.apply(derive, order=0)  # smoothing
        .drop("tail", level=1, axis=1)  # remove tail
        .groupby(level=[0, 2], axis=1)
        .mean()  # get mean x and mean y
    )

    # Calculate frame-wise travelled distance
    travel = (
        pos_centre.diff(axis=0)  # dx, dy
        .pow(2)  # dx^2, dy^2
        .groupby(level=0, axis=1)
        .sum(min_count=2)  # dx^2 + dy^2
        .apply(np.sqrt)  # sqrt
    )

    return travel


def get_body_area_change(pos_df):
    # Calculate rate of change of body area
    delta_areas = (
        pos_df.groupby(level=0, axis=1)
        .apply(
            lambda x: x.iloc[:, :8].agg(
                lambda y: shoelace(y.values.reshape(-1, 2)), axis=1
            )
        )
        .apply(derive, order=1)
        .abs()
    )

    return delta_areas


def get_time_near_source(pos_df, source, factor=1.05):
    # Calculate % of time spent near source
    near_source = (
        pos_df.loc[:, pos_df.columns.get_level_values("bodyparts").isin(("head",))]
        .apply(derive, order=0)  # smoothing
        .groupby(level=[0, 1], axis=1)
        .agg(tuple)  # (x, y) tuple
        .applymap(
            lambda _: is_point_inside_convex_quadrilateral(
                _,
                scale_rectangle(source, factor),
            )
        )
    ).droplevel(1, axis=1)

    return near_source


def build_summary(data, value_name):
    summary_df = (
        pd.concat(data, axis=1)
        .reset_index(names="individuals")
        .melt(id_vars="individuals", value_name=value_name, var_name="video")
    )

    summary_df["camera"] = summary_df["video"].apply(lambda x: x.split("_")[-1])

    summary_df["video_number"] = summary_df.video.apply(
        lambda x: int(x.split("_")[0].replace("GX", ""))
    )

    summary_df["Day"] = (
        summary_df.video_number.sub(
            summary_df.groupby("camera").video_number.transform("min")
        )
        .mul(0.5)
        .add(1)
    )

    return summary_df
