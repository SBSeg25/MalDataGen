#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

# MIT License
#
# Copyright (c) 2025 Synthetic Ocean AI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


try:
    import os
    import sys

    import plotly.io as pio
    from pathlib import Path

    import plotly.graph_objects as go

except ImportError as error:
    print(error)
    sys.exit(-1)

DEFAULT_LOSS_CURVE_LEGEND_GENERATOR = "Generator"
DEFAULT_LOSS_CURVE_LEGEND_DISCRIMINATOR = "Discriminator"
DEFAULT_LOSS_CURVE_LEGEND_ITERATIONS = "Iterations (Epochs)"
DEFAULT_LOSS_CURVE_TITLE_PLOT = "Generator and Discriminator loss"
DEFAULT_LOSS_CURVE_LEGEND_LOSS = "loss"
DEFAULT_LOSS_CURVE_LEGEND_NAME = "Legend"
DEFAULT_LOSS_CURVE_PREFIX_FILE = "curve_training_error"

class PlotCurveLoss:

    def __init__(self, loss_curve_legend_generator=DEFAULT_LOSS_CURVE_LEGEND_GENERATOR,
                 loss_curve_legend_discriminator=DEFAULT_LOSS_CURVE_LEGEND_DISCRIMINATOR,
                 loss_curver_title_plot=DEFAULT_LOSS_CURVE_TITLE_PLOT,
                 loss_curve_legend_iterations=DEFAULT_LOSS_CURVE_LEGEND_ITERATIONS,
                 loss_curve_legend_loss=DEFAULT_LOSS_CURVE_LEGEND_LOSS,
                 loss_curve_legend_name=DEFAULT_LOSS_CURVE_LEGEND_NAME,
                 loss_curve_prefix_file=DEFAULT_LOSS_CURVE_PREFIX_FILE):

        self.loss_curve_legend_generator = loss_curve_legend_generator
        self.loss_curve_legend_discriminator = loss_curve_legend_discriminator
        self.loss_curver_title_plot = loss_curver_title_plot
        self.loss_curve_legend_iterations = loss_curve_legend_iterations
        self.loss_curve_legend_loss = loss_curve_legend_loss
        self.loss_curve_legend_name = loss_curve_legend_name
        self.loss_curve_prefix_file = loss_curve_prefix_file

    def plot_training_loss_curve(self, generator_loss, discriminator_loss, output_dir, k_fold, path_curve_loss):

        if output_dir is not None:
            new_loss_curve_plot = go.Figure()
            new_loss_curve_plot.add_trace(go.Scatter(x=list(range(len(generator_loss))), y=generator_loss,
                                                     name=self.loss_curve_legend_generator))
            new_loss_curve_plot.add_trace(go.Scatter(x=list(range(len(discriminator_loss))), y=discriminator_loss,
                                                     name=self.loss_curve_legend_discriminator))

            new_loss_curve_plot.update_layout(title=self.loss_curver_title_plot,
                                              xaxis_title=self.loss_curve_legend_iterations,
                                              yaxis_title=self.loss_curve_legend_loss,
                                              legend_title=self.loss_curve_legend_name)

            Path(os.path.join(output_dir, path_curve_loss)).mkdir(parents=True, exist_ok=True)
            file_name_output = self.loss_curve_prefix_file + "_k_{}.pdf".format(str(k_fold + 1))
            pio.write_image(new_loss_curve_plot, os.path.join(output_dir, path_curve_loss, file_name_output))

    def set_loss_curve_legend_generator(self, loss_curve_legend_generator):

        self.loss_curve_legend_generator = loss_curve_legend_generator

    def set_loss_curve_legend_discriminator(self, loss_curve_legend_discriminator):

        self.loss_curve_legend_discriminator = loss_curve_legend_discriminator

    def set_loss_curver_title_plot(self, loss_curver_title_plot):

        self.loss_curver_title_plot = loss_curver_title_plot

    def set_loss_curve_legend_iterations(self, loss_curve_legend_iterations):

        self.loss_curve_legend_iterations = loss_curve_legend_iterations

    def set_loss_curve_legend_loss(self, loss_curve_legend_loss):

        self.loss_curve_legend_loss = loss_curve_legend_loss

    def set_loss_curve_legend_name(self, loss_curve_legend_name):

        self.loss_curve_legend_name = loss_curve_legend_name

    def set_loss_curve_prefix_file(self, loss_curve_prefix_file):

        self.loss_curve_prefix_file = loss_curve_prefix_file
