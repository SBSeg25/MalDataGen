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
    import sys
    import numpy
    import logging

    import matplotlib.pyplot as plt
    from sklearn.cluster import Birch
    from sklearn.cluster import OPTICS
    from sklearn.cluster import KMeans
    from sklearn.cluster import DBSCAN
    from sklearn.cluster import MeanShift

    from sklearn.manifold import MDS
    from sklearn.manifold import TSNE
    from sklearn.manifold import Isomap
    from sklearn.decomposition import PCA

    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import AffinityPropagation

    from sklearn.cluster import SpectralClustering
    from sklearn.preprocessing import StandardScaler

    from sklearn.cluster import AgglomerativeClustering
    from sklearn.manifold import LocallyLinearEmbedding


except ImportError as error:
    logging.error(error)
    sys.exit(-1)

try:
    import umap
    UMAP_AVAILABLE = True

except ImportError:
    UMAP_AVAILABLE = False

# Default values for cluster parameters
DEFAULT_CLUSTER_ALGO = 'kmeans'
DEFAULT_REDUCTION_ALGO = 'tsne'
DEFAULT_NUMBER_CLUSTERS = 3
DEFAULT_PERPLEXITY = 30
DEFAULT_NUMBER_COMPONENTS = 3
DEFAULT_RANDOM_STATE = 42
DEFAULT_COLOR_MAP = 'tab10'
DEFAULT_POINT_SIZE = 50
DEFAULT_ALPHA = 0.7
DEFAULT_MARKER_STYLE = 'o'
DEFAULT_THEME = None
DEFAULT_SHOW_LEGEND = True
DEFAULT_TITLE_FONT_SIZE = 16
DEFAULT_LABEL_FONT_SIZE = 12
DEFAULT_LEGEND_FONT_SIZE = 10
DEFAULT_BACKGROUND_COLOR = 'white'
DEFAULT_EDGE_COLOR = 'black'
DEFAULT_LINE_WIDTH = 0.5
DEFAULT_FIGURE_SIZE = (12, 10)
DEFAULT_AX_OFF = False
DEFAULT_GRID = True
DEFAULT_ANNOTATE = False
DEFAULT_CUSTOM_LABELS = None



class ClusteringVisualizer:

    """
    A comprehensive clustering visualization tool that combines dimensionality reduction
    and clustering algorithms to create 2D or 3D visualizations of high-dimensional data.

    This class provides a flexible interface to:
    1. Preprocess data (standard scaling)
    2. Apply various clustering algorithms
    3. Reduce dimensions using different techniques
    4. Visualize results with extensive customization options

    Attributes:
        cluster_algo (str): The clustering algorithm to use (default: 'kmeans')
        reduction_algo (str): The dimensionality reduction algorithm (default: 'tsne')
        number_clusters (int): Number of clusters to form (default: 3)
        perplexity (float): Perplexity parameter for t-SNE (default: 30)
        number_components (int): Number of dimensions for reduction (2 or 3, default: 3)
        random_state (int): Random seed for reproducibility (default: 42)
        color_map (str): Matplotlib colormap name (default: 'tab10')
        point_size (int): Size of scatter plot points (default: 50)
        alpha (float): Transparency of points (0-1, default: 0.7)
        marker_style (str): Marker style for points (default: 'o')
        theme (str): Matplotlib style theme (default: None)
        show_legend (bool): Whether to show legend (default: True)
        title_font_size (int): Title font size (default: 16)
        label_font_size (int): Axis label font size (default: 12)
        legend_font_size (int): Legend font size (default: 10)
        background_color (str): Background color (default: 'white')
        edge_color (str): Point edge color (default: 'black')
        line_width (float): Point edge line width (default: 0.5)
        figure_size (tuple): Figure size in inches (default: (12, 10))
        ax_off (bool): Turn off axis (default: False)
        grid (bool): Show grid (default: True)
        annotate (bool): Annotate points with labels (default: False)
        custom_labels (list): Custom labels for clusters (default: None)

    Example:
    -------
        >>> from sklearn.datasets import make_blobs
        ... X, y = make_blobs(n_samples=300, centers=4, n_features=10, random_state=42)
        ...
        ... # Initialize visualizer with custom parameters
        ... visualizer = ClusteringVisualizer(
        ...     cluster_algo='kmeans',
        ...     reduction_algo='pca',
        ...     number_clusters=4,
        ...     number_components=2,
        ...     point_size=80,
        ...     color_map='viridis',
        ...     theme='ggplot'
        ... )
        ...
        ... # Process and visualize the data
        ... X_scaled = visualizer.preprocess(X)
        ... visualizer.cluster(X_scaled)
        ... visualizer.reduce(X_scaled)
        ... visualizer.plot_clusters()
        ...
        >>> print("Example completed. Check the displayed plot.")
    """

    def __init__(self,
                 cluster_algo=DEFAULT_CLUSTER_ALGO,
                 reduction_algo=DEFAULT_REDUCTION_ALGO,
                 number_clusters=DEFAULT_NUMBER_CLUSTERS,
                 perplexity=DEFAULT_PERPLEXITY,
                 number_components=DEFAULT_NUMBER_COMPONENTS,
                 random_state=DEFAULT_RANDOM_STATE,
                 color_map=DEFAULT_COLOR_MAP,
                 point_size=DEFAULT_POINT_SIZE,
                 alpha=DEFAULT_ALPHA,
                 marker_style=DEFAULT_MARKER_STYLE,
                 theme=DEFAULT_THEME,
                 show_legend=DEFAULT_SHOW_LEGEND,
                 title_font_size=DEFAULT_TITLE_FONT_SIZE,
                 label_font_size=DEFAULT_LABEL_FONT_SIZE,
                 legend_font_size=DEFAULT_LEGEND_FONT_SIZE,
                 background_color=DEFAULT_BACKGROUND_COLOR,
                 edge_color=DEFAULT_EDGE_COLOR,
                 line_width=DEFAULT_LINE_WIDTH,
                 figure_size=DEFAULT_FIGURE_SIZE,
                 ax_off=DEFAULT_AX_OFF,
                 grid=DEFAULT_GRID,
                 annotate=DEFAULT_ANNOTATE,
                 custom_labels=DEFAULT_CUSTOM_LABELS,
                 export_path=''):
        """
        Initialize the ClusteringVisualizer with specified parameters.

        Args:
            See class attributes for parameter descriptions.
        """
        self.cluster_algo = cluster_algo.lower()
        self.reduction_algo = reduction_algo.lower()
        self.number_clusters = number_clusters
        self.perplexity = perplexity
        self.number_components = number_components
        self.random_state = random_state

        # Visualization parameters
        self.color_map = color_map
        self.point_size = point_size
        self.alpha = alpha
        self.marker_style = marker_style
        self.theme = theme
        self.show_legend = show_legend
        self.title_font_size = title_font_size
        self.label_font_size = label_font_size
        self.legend_font_size = legend_font_size
        self.background_color = background_color
        self.edge_color = edge_color
        self.linewidth = line_width
        self.figure_size = figure_size
        self.ax_off = ax_off
        self.grid = grid
        self.export_path = export_path
        self.annotate = annotate
        self.custom_labels = custom_labels

        # Internal objects
        self.scaler = StandardScaler()
        self.labels = None
        self.X_embedded = None

        # Validate parameters and initialize models
        self._validate_parameters()
        self.cluster_model = self._init_cluster_model()
        self.reducer_model = self._init_reduction_model()

    def _validate_parameters(self):
        """
        Validate input parameters to ensure they are within acceptable ranges.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if not isinstance(self.number_clusters, int) or self.number_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer.")

        if not isinstance(self.perplexity, (int, float)) or self.perplexity <= 0:
            raise ValueError("perplexity must be a positive number.")

        if self.number_components not in [2, 3]:
            raise ValueError("n_components must be either 2 or 3.")

    def _init_cluster_model(self):
        """
        Initialize the clustering model based on specified algorithm.

        Returns:
            The initialized clustering model.

        Raises:
            ValueError: If the specified clustering algorithm is not supported.
        """
        models = {
            'kmeans': lambda: KMeans(n_clusters=self.number_clusters, random_state=self.random_state),
            'dbscan': DBSCAN,
            'agglo': lambda: AgglomerativeClustering(n_clusters=self.number_clusters),
            'spectral': lambda: SpectralClustering(n_clusters=self.number_clusters,
                                                   random_state=self.random_state, affinity='nearest_neighbors'),
            'meanshift': MeanShift,
            'birch': lambda: Birch(n_clusters=self.number_clusters),
            'optics': OPTICS,
            'gaussian': lambda: GaussianMixture(n_components=self.number_clusters, random_state=self.random_state),
            'affinity': lambda: AffinityPropagation(random_state=self.random_state)
        }

        if self.cluster_algo not in models:
            raise ValueError(f"Unsupported clustering algorithm: '{self.cluster_algo}'.")

        return models[self.cluster_algo]()

    def _init_reduction_model(self):
        """
        Initialize the dimensionality reduction model based on specified algorithm.

        Returns:
            The initialized reduction model.

        Raises:
            ValueError: If the specified reduction algorithm is not supported.
            ImportError: If UMAP is requested but not installed.
        """
        models = {
            'tsne': lambda: TSNE(n_components=self.number_components,
                                 perplexity=self.perplexity, random_state=self.random_state),
            'pca': lambda: PCA(n_components=self.number_components),
            'svd': lambda: TruncatedSVD(n_components=self.number_components),
            'mds': lambda: MDS(n_components=self.number_components, random_state=self.random_state),
            'isomap': lambda: Isomap(n_components=self.number_components),
            'lle': lambda: LocallyLinearEmbedding(n_components=self.number_components),
            'umap': lambda: umap.UMAP(n_components=self.number_components, random_state=self.random_state)
        }

        if self.reduction_algo == 'umap' and not UMAP_AVAILABLE:
            raise ImportError("UMAP is not installed. Use: pip install umap-learn")

        if self.reduction_algo not in models:
            raise ValueError(f"Unsupported dimensionality reduction algorithm: '{self.reduction_algo}'.")

        return models[self.reduction_algo]()

    def preprocess(self, data: numpy.ndarray) -> numpy.ndarray:
        """
        Preprocess the input data by standard scaling (mean=0, variance=1).

        Args:
            data: Input data as a NumPy array (n_samples, n_features)

        Returns:
            Scaled data as a NumPy array.

        Raises:
            ValueError: If input data is not a NumPy array.
        """
        if not isinstance(data, numpy.ndarray):
            raise ValueError("Input data must be a NumPy array.")
        data = self.scaler.fit_transform(data)
        self.reduce(data)
        return data

    def cluster(self, x_scaled: numpy.ndarray) -> numpy.ndarray:
        """
        Apply clustering to the scaled data.

        Args:
            x_scaled: Preprocessed data (output from preprocess())

        Returns:
            Cluster labels as a NumPy array.
        """
        if self.cluster_algo == 'gaussian':
            self.labels = self.cluster_model.fit(x_scaled).predict(x_scaled)
        else:
            self.labels = self.cluster_model.fit_predict(x_scaled)

        return self.labels

    def reduce(self, x_scaled: numpy.ndarray) -> numpy.ndarray:
        """
        Apply dimensionality reduction to the scaled data.

        Args:
            x_scaled: Preprocessed data (output from preprocess())

        Returns:
            Reduced data as a NumPy array with shape (n_samples, n_components)
        """
        self.X_embedded = self.reducer_model.fit_transform(x_scaled)
        return self.X_embedded

    def plot_clusters(self):
        """
        Create and display the cluster visualization plot.

        Raises:
            ValueError: If reduce() and cluster() haven't been called first.
        """
        self._check_plot_ready()
        self._apply_theme(self.theme)
        self._plot()

    def _check_plot_ready(self):
        """Check if required data is available for plotting."""
        if self.X_embedded is None or self.labels is None:
            raise ValueError("You must run reduce() and cluster() before plotting.")

    @staticmethod
    def _apply_theme(theme):
        """
        Apply a matplotlib style theme if specified.

        Args:
            theme: Name of matplotlib style to apply

        Raises:
            ValueError: If theme name is invalid.
        """
        if theme:
            try:
                plt.style.use(theme)
            except OSError:
                raise ValueError(f"Invalid theme name: {theme}")

    def _plot(self):
        """Main method to create the cluster visualization plot."""
        is_3d = self.number_components == 3
        fig, ax = self._setup_figure(is_3d)
        self._plot_clusters(fig, ax, is_3d)
        self._configure_axes(ax, is_3d)
        self._finalize_plot()

    def _setup_figure(self, is_3d):
        """Initialize the figure and axes with proper settings."""
        fig = plt.figure(figsize=self.figure_size)
        ax = fig.add_subplot(111, projection='3d' if is_3d else None)
        fig.patch.set_facecolor(self.background_color)
        ax.set_facecolor(self.background_color)
        return fig, ax

    def _get_cluster_colors(self):
        """Generate distinct colors for each cluster using the specified colormap."""
        cmap = plt.get_cmap(self.color_map)
        unique_labels = numpy.unique(self.labels)
        num_clusters = len(unique_labels)
        return [cmap(i / num_clusters) for i in range(num_clusters)], unique_labels

    def _plot_clusters(self, fig, ax, is_3d):
        """Plot each cluster with its distinct color and proper styling."""
        colors, unique_labels = self._get_cluster_colors()

        for i, cluster in enumerate(unique_labels):
            self._plot_single_cluster(ax, cluster, colors[i], is_3d)

    def _plot_single_cluster(self, ax, cluster, color, is_3d):
        """Plot a single cluster with all its points."""
        mask = self.labels == cluster
        coordinates = self.X_embedded[mask]
        label = self._get_cluster_label(cluster, len(coordinates))

        scatter_kwargs = {
            's': self.point_size,
            'alpha': self.alpha,
            'marker': self.marker_style,
            'edgecolors': self.edge_color,
            'linewidths': self.linewidth,
            'label': label,
            'color': color
        }

        if is_3d:
            ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], **scatter_kwargs)
        else:
            ax.scatter(coordinates[:, 0], coordinates[:, 1], **scatter_kwargs)

        if self.annotate:
            self._annotate_points(ax, coordinates, label, is_3d)

    def _get_cluster_label(self, cluster, cluster_size):
        """Generate the appropriate label for a cluster."""
        if self.custom_labels and cluster < len(self.custom_labels):
            return self.custom_labels[cluster]
        return f'Cluster {cluster} (n={cluster_size})'

    @staticmethod
    def _annotate_points(ax, coordinates, label, is_3d):
        """Add text annotations to each point if enabled."""
        for x, y, *z in coordinates:
            if is_3d:
                ax.text(x, y, z[0] if z else 0, str(label), fontsize=8)
            else:
                ax.text(x, y, str(label), fontsize=8)

    def _configure_axes(self, ax, is_3d):
        """Configure axis labels, title, and other properties."""
        ax.set_title(
            f"Clusters ({self.reduction_algo.upper()} {self.number_components}D + {self.cluster_algo.upper()})",
            fontsize=self.title_font_size)
        ax.set_xlabel("Component 1", fontsize=self.label_font_size)
        ax.set_ylabel("Component 2", fontsize=self.label_font_size)

        if is_3d:
            ax.set_zlabel("Component 3", fontsize=self.label_font_size)

        if self.ax_off:
            ax.axis('off')
        else:
            ax.grid(self.grid)

        if self.show_legend:
            ax.legend(fontsize=self.legend_font_size)

    def _finalize_plot(self):
        """Apply final adjustments and display the plot."""
        plt.tight_layout()
        plt.savefig(self.export_path, format="pdf")

    @property
    def cluster_algo(self) -> str:
        """Get the clustering algorithm name in lowercase.

        Returns:
            str: Name of the clustering algorithm in lowercase.
        """
        return self._cluster_algo

    @cluster_algo.setter
    def cluster_algo(self, value: str) -> None:
        """Set the clustering algorithm name (converted to lowercase).

        Args:
            value (str): Name of the clustering algorithm.

        Raises:
            TypeError: If input is not a string.
        """
        if not isinstance(value, str):
            raise TypeError("cluster_algo must be a string")
        self._cluster_algo = value.lower()

    @property
    def reduction_algo(self) -> str:
        """Get the dimensionality reduction algorithm name in lowercase.

        Returns:
            str: Name of the reduction algorithm in lowercase.
        """
        return self._reduction_algo

    @reduction_algo.setter
    def reduction_algo(self, value: str) -> None:
        """Set the dimensionality reduction algorithm name (converted to lowercase).

        Args:
            value (str): Name of the reduction algorithm.

        Raises:
            TypeError: If input is not a string.
        """
        if not isinstance(value, str):
            raise TypeError("reduction_algo must be a string")
        self._reduction_algo = value.lower()

    @property
    def number_clusters(self) -> int:
        """Get the number of clusters to generate.

        Returns:
            int: Number of clusters.
        """
        return self._number_clusters

    @number_clusters.setter
    def number_clusters(self, value: int) -> None:
        """Set the number of clusters to generate.

        Args:
            value (int): Number of clusters (must be positive).

        Raises:
            TypeError: If input is not an integer.
            ValueError: If input is not positive.
        """
        if not isinstance(value, int):
            raise TypeError("number_clusters must be an integer")
        if value <= 0:
            raise ValueError("number_clusters must be positive")
        self._number_clusters = value

    @property
    def perplexity(self) -> float:
        """Get the perplexity parameter for t-SNE.

        Returns:
            float: Perplexity value.
        """
        return self._perplexity

    @perplexity.setter
    def perplexity(self, value: float) -> None:
        """Set the perplexity parameter for t-SNE.

        Args:
            value (float): Perplexity value (should be positive).

        Raises:
            TypeError: If input is not a number.
            ValueError: If input is not positive.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("perplexity must be a number")
        if value <= 0:
            raise ValueError("perplexity must be positive")
        self._perplexity = float(value)

    @property
    def number_components(self) -> int:
        """Get the number of components for dimensionality reduction.

        Returns:
            int: Number of components.
        """
        return self._number_components

    @number_components.setter
    def number_components(self, value: int) -> None:
        """Set the number of components for dimensionality reduction.

        Args:
            value (int): Number of components (must be 2 or 3 for visualization).

        Raises:
            TypeError: If input is not an integer.
            ValueError: If input is not 2 or 3.
        """
        if not isinstance(value, int):
            raise TypeError("number_components must be an integer")
        if value not in (2, 3):
            raise ValueError("number_components must be 2 or 3 for visualization")
        self._number_components = value

    @property
    def random_state(self) -> int:
        """Get the random seed for reproducibility.

        Returns:
            int: Random seed value.
        """
        return self._random_state

    @random_state.setter
    def random_state(self, value: int) -> None:
        """Set the random seed for reproducibility.

        Args:
            value (int): Random seed value.

        Raises:
            TypeError: If input is not an integer.
        """
        if not isinstance(value, int):
            raise TypeError("random_state must be an integer")
        self._random_state = value

    # Visualization parameters
    @property
    def color_map(self) -> str:
        """Get the colormap name for visualization.

        Returns:
            str: Name of the colormap.
        """
        return self._color_map

    @color_map.setter
    def color_map(self, value: str) -> None:
        """Set the colormap for visualization.

        Args:
            value (str): Name of a valid matplotlib colormap.

        Raises:
            TypeError: If input is not a string.
        """
        if not isinstance(value, str):
            raise TypeError("color_map must be a string")
        self._color_map = value

    @property
    def point_size(self) -> float:
        """Get the point size for scatter plots.

        Returns:
            float: Size of points in the plot.
        """
        return self._point_size

    @point_size.setter
    def point_size(self, value: float) -> None:
        """Set the point size for scatter plots.

        Args:
            value (float): Size of points (must be positive).

        Raises:
            TypeError: If input is not a number.
            ValueError: If input is not positive.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("point_size must be a number")
        if value <= 0:
            raise ValueError("point_size must be positive")
        self._point_size = float(value)

    # Cluster parameters
    @cluster_algo.setter
    def cluster_algo(self, value):
        self._cluster_algo = value.lower()

    @reduction_algo.setter
    def reduction_algo(self, value):
        self._reduction_algo = value.lower()

    @number_clusters.setter
    def number_clusters(self, value):
        self._number_clusters = value

    @perplexity.setter
    def perplexity(self, value):
        self._perplexity = value


    @number_components.setter
    def number_components(self, value):
        self._number_components = value


    @random_state.setter
    def random_state(self, value):
        self._random_state = value

    @color_map.setter
    def color_map(self, value):
        self._color_map = value


    @point_size.setter
    def point_size(self, value):
        self._point_size = value

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def marker_style(self):
        return self._marker_style

    @marker_style.setter
    def marker_style(self, value):
        self._marker_style = value

    @property
    def theme(self):
        return self._theme

    @theme.setter
    def theme(self, value):
        self._theme = value

    @property
    def show_legend(self):
        return self._show_legend

    @show_legend.setter
    def show_legend(self, value):
        self._show_legend = value

    @property
    def title_font_size(self):
        return self._title_font_size

    @title_font_size.setter
    def title_font_size(self, value):
        self._title_font_size = value

    @property
    def label_font_size(self):
        return self._label_font_size

    @label_font_size.setter
    def label_font_size(self, value):
        self._label_font_size = value

    @property
    def legend_font_size(self):
        return self._legend_font_size

    @legend_font_size.setter
    def legend_font_size(self, value):
        self._legend_font_size = value

    @property
    def background_color(self):
        return self._bg_color

    @background_color.setter
    def background_color(self, value):
        self._bg_color = value

    @property
    def edge_color(self):
        return self._edge_color

    @edge_color.setter
    def edge_color(self, value):
        self._edge_color = value

    @property
    def linewidth(self):
        return self._linewidth

    @linewidth.setter
    def linewidth(self, value):
        self._linewidth = value

    @property
    def figure_size(self):
        return self._figure_size

    @figure_size.setter
    def figure_size(self, value):
        self._figure_size = value

    @property
    def ax_off(self):
        return self._ax_off

    @ax_off.setter
    def ax_off(self, value):
        self._ax_off = value

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, value):
        self._grid = value

    @property
    def annotate(self):
        return self._annotate

    @annotate.setter
    def annotate(self, value):
        self._annotate = value

    @property
    def custom_labels(self):
        return self._custom_labels

    @custom_labels.setter
    def custom_labels(self, value):
        self._custom_labels = value
