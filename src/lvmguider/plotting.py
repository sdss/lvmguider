#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-09-03
# @Filename: plotting.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import os
import pathlib

from typing import TYPE_CHECKING, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy
import seaborn

from lvmguider import config


if TYPE_CHECKING:
    import pandas
    from matplotlib.axes import Axes

    from lvmguider.dataclasses import GlobalSolution


matplotlib.use("agg")


AnyPath = str | os.PathLike


def plot_qa(
    solution: GlobalSolution,
    root: AnyPath | None = None,
    save_subplots: bool = True,
):
    """Produces guider QA plots from a global solution object."""

    root = pathlib.Path(root or (solution.path.parent / "qa"))
    root.mkdir(parents=True, exist_ok=True)

    stem = solution.path.stem

    # Set up seaborn and matplotlib.
    seaborn.set_theme(style="darkgrid", palette="deep", font="serif")

    with plt.ioff():
        # Plot PA
        outpath_pa = root / (stem + "_pa.pdf")
        plot_position_angle(outpath_pa, solution, save_subplots=save_subplots)

        # Plot ZP
        outpath_zp = root / (stem + "_zp.pdf")
        plot_zero_point_or_fwhm(
            outpath_zp,
            solution,
            save_subplots=save_subplots,
            column="zero_point",
        )

        # Plot FWHM
        outpath_fwhm = root / (stem + "_fwhm.pdf")
        plot_zero_point_or_fwhm(
            outpath_fwhm,
            solution,
            save_subplots=save_subplots,
            column="fwhm",
        )

    seaborn.reset_orig()


def create_subplot_path(
    orig_path: AnyPath,
    suffix: str = "",
    extension="png",
    subdir: str = "subplots",
):
    """Creates a path for a subplot."""

    orig_path = pathlib.Path(orig_path)

    orig_parent = orig_path.parent
    stem = orig_path.stem

    path = orig_parent

    if subdir:
        path /= subdir

    return path / (stem + suffix + f".{extension}")


def create_subplot(func: Callable, path: AnyPath, *args, dpi: int = 100, **kwargs):
    """Creates a subplot by calling a plotting function."""

    fig, ax = plt.subplots(figsize=(11, 8))
    func(ax, *args, **kwargs)
    fig.savefig(str(path), dpi=dpi)


def get_figure():
    """Returns a customised figure and axes."""

    fig, axd = plt.subplot_mosaic(
        [["east", "west"], ["global", "global"]],
        figsize=(11, 8),
    )
    fig.subplots_adjust(
        left=0.1,
        right=0.9,
        bottom=0.075,
        top=0.91,
        wspace=0.4,
        hspace=0.25,
    )

    for name, ax in axd.items():
        ax.axis("off")

        ax.set_xlabel("Frame number")

        if name == "global":
            ax.set_title("Full frame")
        else:
            ax.set_title(f"Camera {str(name).capitalize()}")

        ax.ticklabel_format(useOffset=False)
        ax.xaxis.get_major_locator().set_params(integer=True)

    return fig, axd


def plot_position_angle(
    outpath: pathlib.Path,
    solution: GlobalSolution,
    save_subplots: bool = False,
):
    """Plots the position angle."""

    fig, axd = get_figure()

    # Plot top panels with the East and West camera PA and PA error.
    for camera_solution in solution.coadd_solutions:
        if camera_solution.frames is None or len(camera_solution.frames) == 0:
            continue

        camera = camera_solution.camera
        frame_data = camera_solution.frame_data()

        if len(frame_data.pa.dropna()) < 2:
            continue

        # Unhide plot.
        ax = axd[camera]  # type: ignore
        ax.axis("on")

        # Plot PA.
        axes = _plot_pa_axes(
            ax,
            frame_data,
            pa_error_mode="mean",
            legend=False,
        )

        # Remove the inside labels since they are identical left and right.
        if camera == "east":
            axes[1].set_ylabel("")
        else:
            axes[0].set_ylabel("")

        if save_subplots:
            create_subplot(
                _plot_pa_axes,
                create_subplot_path(outpath, f"_{camera}"),
                frame_data,
                title=f"Camera {camera.capitalize()}",
                pa_error_mode="mean",
                legend=True,
            )

    # Now use the PAs from the guider data (i.e., full frame).
    guider_data = solution.guider_data()
    guider_data = guider_data.loc[guider_data.guide_mode == "guide"]
    if len(guider_data.pa.dropna()) >= 2:
        ax = axd["global"]  # type: ignore
        ax.axis("on")

        _plot_pa_axes(ax, guider_data, pa_error_mode="first", legend=True)

        if save_subplots:
            create_subplot(
                _plot_pa_axes,
                create_subplot_path(outpath),
                guider_data,
                title="Full frame",
                pa_error_mode="first",
                legend=True,
            )

    fig.suptitle(f"Position angle for {solution.path.name}")
    fig.savefig(str(outpath))

    plt.close("all")


def _plot_pa_axes(
    ax: Axes,
    data: pandas.DataFrame,
    title: str = "",
    pa_error_mode: str = "first",
    legend: bool = True,
):
    """Plot PA axes."""

    pa_threshold = config["coadds"]["warnings"]["pa_error"]

    # Plot absolute position angle.
    (pa_plot,) = ax.plot(
        data.frameno,
        data.pa,
        "b-",
        zorder=20,
        label="Position angle",
    )

    # Create right axis for the PA error.
    right_ax = ax.twinx()
    right_ax.grid(False)

    # For the cameras we calculate the error wrt the mean PA.
    # For the full frame PAs we use the initial PA and calculate the "drift".
    if pa_error_mode == "first":
        ref_pa = data.pa.dropna().iloc[0]
    else:
        ref_pa = data.pa.dropna().mean()

    # Plot PA error.
    (pa_error_plot,) = right_ax.plot(
        data.frameno,
        ref_pa - data.pa,
        "g-",
        zorder=20,
        label="PA error",
    )

    # If the PA error goes over the limit, show a band with the threshold.
    abs_error = (ref_pa - data.pa).abs().max()
    if abs_error > pa_threshold:
        right_ax.axhspan(
            ymin=-0.0025,
            ymax=0.0025,
            facecolor="red",
            edgecolor="None",
            linestyle="None",
            alpha=0.2,
            zorder=5,
        )

    if legend:
        right_ax.legend(
            handles=[pa_plot, pa_error_plot],
            loc="upper left",
        ).set_zorder(100)
        ax.margins(0.05, 0.2)

    ax.set_xlabel("Frame number")
    ax.set_ylabel("Position angle [deg]", labelpad=10)
    right_ax.set_ylabel("Position angle error [deg]", labelpad=10)

    if title:
        ax.set_title(title)

    return [ax, right_ax]


def plot_zero_point_or_fwhm(
    outpath: pathlib.Path,
    solution: GlobalSolution,
    save_subplots: bool = False,
    column="zero_point",
):
    """Plots the zero point or FWHM of an exposure."""

    fig, axd = get_figure()

    # Plot top panels with the East and West camera zero point or FWHM.
    for camera_solution in solution.coadd_solutions:
        camera = camera_solution.camera

        # Get all the frames and the ZP/FWHM value of the co-added camera image.
        frame_data = camera_solution.frame_data()
        coadd_value = getattr(camera_solution, column)

        if len(frame_data) == 0 and numpy.isnan(coadd_value):
            continue

        # Un-hide the axes.
        ax = axd[camera]  # type: ignore
        ax.axis("on")

        # Plot data.
        ax = _plot_zero_point_or_fwhm_axes(
            ax,
            frame_data,
            coadd_value,
            column=column,
            legend=False,
        )

        # Move the West camera y-axis to the right to unclutter the central section.
        if camera == "west":
            ax.yaxis.set_label_position("right")

        if save_subplots:
            create_subplot(
                _plot_zero_point_or_fwhm_axes,
                create_subplot_path(outpath, f"_{camera}"),
                frame_data,
                coadd_value,
                column=column,
                title=f"Camera {camera.capitalize()}",
                legend=True,
            )

    # Now do the full frame. Almost identical to above.
    guider_data = solution.guider_data()
    guider_data = guider_data.loc[guider_data.guide_mode == "guide"]
    global_value = getattr(solution, column)

    if len(guider_data.zero_point.dropna()) >= 2 or not numpy.isnan(global_value):
        ax = axd["global"]  # type: ignore
        ax.axis("on")

        _plot_zero_point_or_fwhm_axes(
            ax,
            guider_data,
            global_value,
            column=column,
            legend=True,
        )

        if save_subplots:
            create_subplot(
                _plot_zero_point_or_fwhm_axes,
                create_subplot_path(outpath),
                guider_data,
                global_value,
                column=column,
                title="Full frame",
                legend=True,
            )

    if column == "zero_point":
        fig.suptitle(f"Zero point for {solution.path.name}")
    else:
        fig.suptitle(f"FWHM for {solution.path.name}")

    fig.savefig(str(outpath))

    plt.close("all")


def _plot_zero_point_or_fwhm_axes(
    ax: Axes,
    data: pandas.DataFrame,
    coadd: float,
    column="zero_point",
    title: str | None = None,
    legend: bool = True,
):
    """Plot zero point or FWHM axes."""

    handles = []

    # Plot the data for each frame.
    (plot,) = ax.plot(
        data.frameno,
        data[column],
        "b-",
        label="Frames",
    )
    handles.append(plot)

    # If the co-added value exists, plot it as a horizontal dashed line.
    if not numpy.isnan(coadd):
        median_line = ax.axhline(
            y=coadd,
            color="k",
            linestyle="dashed",
            label="Co-add",
            alpha=0.5,
        )
        handles.append(median_line)

    if legend:
        loc = "lower right" if column == "zero_point" else "upper left"
        ax.legend(handles=handles, loc=loc).set_zorder(100)
        ax.margins(0.05, 0.25)

    ax.set_xlabel("Frame number")

    if column == "zero_point":
        ax.set_ylabel("Zero Point [mag]", labelpad=10)
    else:
        ax.set_ylabel("FWHM [arcsec]", labelpad=10)

    if title:
        ax.set_title(title)

    return ax
