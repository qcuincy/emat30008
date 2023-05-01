import plotly.graph_objs as go
from plotly.subplots import make_subplots
from tabulate import tabulate
from art import *
import matplotlib.cm as cm
import numpy as np

class Solution():
    """
    Solution class for storing solution data

    Classes:
        Solution()

    Attributes:
        t (ndarray): Array of time values
        y (ndarray): Array of solution values
        u (ndarray): Array of solution values for PDEs
        problem_type (str): "ODE" or "PDE"
        shape (dict): Dictionary of shapes of t, y and u

    Methods:
        plot(width=800, height=600, title="", exact=None):
            Plot the solution

    Usage:
        >>> from numerical_methods.solution import Solution

        >>> t = np.linspace(0, 10, 100)
        >>> y = np.sin(t)
        >>> sol = Solution(t, y)
        >>> sol.plot()
    """
    def __init__(self, t, y, u=None, params=None):
        self.t = t
        self.y = y
        self.u = u
        self.params = params
        self.problem_type = "PDE" if u is not None else "ODE" if params is None else "CONTINUATION"
        self.shape = self.__shape__()

    def __shape__(self):
        return dict(t=self.t.shape, y=self.y.shape, u=self.u.shape) if self.problem_type == "PDE" else dict(t=self.t.shape, y=self.y.shape) if self.problem_type == "ODE" else dict(t=self.t.shape, params=self.params.shape, y=self.y.shape)
    
    def __repr__(self):
        if self.problem_type == "PDE":
            data = [["t",self.t, self.t.shape], ["y",self.y, self.y.shape], ["u",self.u, self.u.shape]]
        elif self.problem_type == "ODE":
            data = [["t",self.t, self.t.shape], ["y",self.y, self.y.shape]]
        else:
            data = [["t",self.t, self.t.shape], ["parameters",self.params, self.params.shape], ["y",self.y, self.y.shape]]

        return text2art(self.problem_type + " Solution", font="small") + tabulate(data,tablefmt="fancy_grid",headers=['Parameter', 'Value', 'Shape'])


    def __len__(self):
        return len(self.t)
    

    def plot(self, width=600, height=400, margin=None, title="", xaxis_title="t", yaxis_title="y(t)", z_axis_title="u(t,y)", exact=None, phase_plot=False, problem_type=None):
        """
        Plot the solution

        Args:
            width (int): Width of the plot
            height (int): Height of the plot
            title (str): Title of the plot
            exact (Solution): Exact solution to be plotted
            phase_plot (bool): Plot phase plot if True

        Returns:
            Figure: Plotly figure object

        Raises:
            TypeError: If exact is not of type Solution or callable with signature exact(t, y)
        """
        fig = go.Figure()
        self.problem_type = problem_type if problem_type is not None else self.problem_type
        if self.problem_type == 'ODE':
            if phase_plot:
                # title = "ODE Phase Plot" if title == "" else title
                solution_trace = []
                for i in range(self.y.shape[1]):
                    solution_trace.append(go.Scatter(x=self.t, y=self.y[:, i], mode="lines", name=f"Solution {i}"))
                # create phase plot trace
                phase_trace = []
                for i in range(self.y.shape[1]-1):
                    phase_trace.append(go.Scatter(x=self.y[:, i], y=self.y[:, i+1], mode="lines", name=f"Phase Plot {i}"))
                # add exact solution trace
                if exact is not None:
                    if callable(exact):
                        exact = Solution(self.t, exact(self.t, self.y))
                    elif isinstance(exact, Solution):
                        exact = go.Scatter(x=exact.y[:, 0], y=exact.y[:, 1], mode="lines", name="Exact Solution", showscale=False)
                    else:
                        raise TypeError("Exact solution must be of type Solution or callable with signature exact(t, y).")
                    phase_trace.append(exact)
                # create ODE solution plot trace
                # add exact solution trace
                if exact is not None:
                    if callable(exact):
                        exact = Solution(self.t, exact(self.t, self.y))
                    elif isinstance(exact, Solution):
                        exact = go.Scatter(x=exact.t, y=exact.y, mode="lines", name="Exact Solution", showscale=False)
                    else:
                        raise TypeError("Exact solution must be of type Solution or callable with signature exact(t, y).")
                    solution_trace.append(exact)
                # create subplots
                fig = make_subplots(rows=1, cols=2, subplot_titles=("ODE Solution Plot", "ODE Phase Plot"))
                fig.add_traces(phase_trace, rows=1, cols=2)
                fig.add_traces(solution_trace, rows=1, cols=1)
                # update x and y axis titles
                fig.update_xaxes(title_text="y1", row=1, col=2)
                fig.update_xaxes(title_text="t", row=1, col=1)
                fig.update_yaxes(title_text="y2", row=1, col=2)
                fig.update_yaxes(title_text="y(t)", row=1, col=1)
                fig.update_layout(
                    width=width,
                    height=height,
                    title=title,
                    margin=dict(l=50, r=50, b=100, t=100, pad=4) if margin is None else margin,
                )
            else:
                title = "ODE Solution Plot" if title == "" else title
                xaxis_title = "t" if xaxis_title == "t" else xaxis_title
                yaxis_title = "y(t)" if yaxis_title == "y(t)" else yaxis_title
                # plot solution plot
                for i in range(self.y.shape[1]):
                    fig.add_trace(go.Scatter(x=self.t, y=self.y[:, i], mode="lines", name=f"Solution {i}"))
                # Check if exact is provided and is type Solution
                if exact is not None:
                    if callable(exact):
                        exact = Solution(self.t, exact(self.t, self.y))
                    elif isinstance(exact, Solution):
                        fig.add_trace(go.Scatter(x=exact.t, y=exact.y, mode="lines", name="Exact Solution", showscale=False))
                    else:
                        raise TypeError("Exact solution must be of type Solution or callable with signature exact(t, y).")
                # Check if min(t) and max(t) are not equal
                if np.nanmin(self.t) != np.nanmax(self.t):
                    x_tickvals = np.arange(np.nanmin(self.t), np.nanmax(self.t), (np.nanmax(self.t) - np.nanmin(self.t)) / 5) # tick values
                else:
                    x_tickvals = np.ones(5) * np.nanmin(self.t)
                x_ticktext = [f"{x:.2f}" for x in x_tickvals] # tick text
                # Check if min(y) and max(y) are not equal
                if np.nanmin(self.y) != np.nanmax(self.y):
                    y_tickvals = np.arange(np.nanmin(self.y), np.nanmax(self.y), (np.nanmax(self.y) - np.nanmin(self.y)) / 5)
                else:
                    y_tickvals = np.ones(5) * np.nanmin(self.y)
                y_ticktext = [f"{y:.2f}" for y in y_tickvals] # tick text
                fig.update_traces(hovertemplate='t: %{x}<br>y: %{y}')
                fig.update_layout(
                    title_text=title, 
                    width=width, 
                    height=height, 
                    scene=dict(xaxis_title=xaxis_title, yaxis_title=yaxis_title), 
                    xaxis=dict(tickmode='array', tickvals=x_tickvals, ticktext=x_ticktext), 
                    yaxis=dict(tickmode='array', tickvals=y_tickvals, ticktext=y_ticktext),
                    margin=dict(l=50, r=50, b=100, t=100, pad=4) if margin is None else margin,
                    )
        elif self.problem_type == 'PDE':
            title = "PDE Solution Plot" if title == "" else title
            # plot surface plot
            fig.add_trace(go.Surface(x=self.t, y=self.y, z=self.u.T, name="Solution", showlegend=True, showscale=False))
            # Check if exact is provided and is type Solution
            if exact is not None:
                if callable(exact):
                    T, Y = np.meshgrid(self.t, self.y)
                    exact = Solution(self.t, self.y, exact(T, Y))
                
                if isinstance(exact, Solution):
                    fig.add_trace(go.Surface(x=exact.t, y=exact.y, z=exact.u.T, name='Exact Solution', showlegend=True, showscale=False))
                else:
                    raise TypeError("Exact solution must be of type Solution or callable with signature exact(t, y).")
            # Check if min(t) and max(t) are the same
            if np.nanmin(self.t) != np.nanmax(self.t):
                x_tickvals = np.arange(np.nanmin(self.t), np.nanmax(self.t), (np.nanmax(self.t) - np.nanmin(self.t)) / 5) # tick values
            else:
                x_tickvals = np.ones(5) * np.nanmin(self.t)
            x_ticktext = [f"{x:.2f}" for x in x_tickvals] # tick text
            # Check if min(y) and max(y) are the same
            if np.nanmin(self.y) != np.nanmax(self.y):
                y_tickvals = np.arange(np.nanmin(self.y), np.nanmax(self.y), (np.nanmax(self.y) - np.nanmin(self.y)) / 5)
            else:
                y_tickvals = np.ones(5) * np.nanmin(self.y)
            y_ticktext = [f"{y:.2f}" for y in y_tickvals] # tick text
            fig.update_traces(hovertemplate='t: %{x}<br>y: %{y}<br>u: %{z}')
            fig.update_layout(
                title_text=title, 
                width=width, 
                height=height, 
                scene=dict(xaxis_title=xaxis_title, yaxis_title=yaxis_title, zaxis_title=z_axis_title),
                xaxis=dict(tickmode='array', tickvals=x_tickvals, ticktext=x_ticktext),
                yaxis=dict(tickmode='array', tickvals=y_tickvals, ticktext=y_ticktext),
                margin=dict(l=50, r=50, b=50, t=50, pad=0) if margin is None else margin,
                )
        
        elif self.problem_type == 'CONTINUATION':
            title = "Continuation Solution Plot" if title == "" else title
            xaxis_title = "param" if xaxis_title == "param" else xaxis_title
            # plot solution plot
            Ns = self.params.shape[0]
            color_map = self.make_color_map(self.params)

            for i in range(Ns):
                color = f'rgb({int(color_map[i][0]*255)}, {int(color_map[i][1]*255)}, {int(color_map[i][2]*255)})'
                solution_val =  self.y[:, i]  # assuming solution_values is Nt x Ns
                time_val = self.t[:len(solution_val)]
                for j in range(solution_val.shape[-1]):
                    # fig.add_trace(go.Scatter(x=time_val, y=solution_val[:, j], name=f"y{i+1}_{j+1}", showlegend=True))
                    fig.add_trace(go.Scatter(x=time_val, y=solution_val[:, j], name=f"p{i+1}, y{j+1}", line=dict(color=color), showlegend=True))
            fig.update_traces(hovertemplate='param: %{x}<br>t:%{x}<br>y: %{y}')
            # Check if min(x) and max(x) are not equal
            if np.nanmin(self.params) != np.nanmax(self.params):
                x_tickvals = np.arange(np.nanmin(self.params), np.nanmax(self.params), (np.nanmax(self.params) - np.nanmin(self.params)) / 5)
            else:
                x_tickvals = np.ones(5) * np.nanmin(self.params)
            x_ticktext = [f"{x:.2f}" for x in x_tickvals] # tick text
            # Check if min(y) and max(y) are not equal
            if np.nanmin(self.y) != np.nanmax(self.y):
                y_tickvals = np.arange(np.nanmin(self.y), np.nanmax(self.y), (np.nanmax(self.y) - np.nanmin(self.y)) / 5)
            else:
                y_tickvals = np.ones(5) * np.nanmin(self.y)
            y_ticktext = [f"{y:.2f}" for y in y_tickvals] # tick text
            fig.update_layout(
                title_text=title,
                width=width,
                height=height,
                xaxis=dict(title=xaxis_title, tickmode='array', tickvals=x_tickvals, ticktext=x_ticktext),
                yaxis=dict(title=yaxis_title, tickmode='array', tickvals=y_tickvals, ticktext=y_ticktext),
                margin=dict(l=50, r=50, b=50, t=50, pad=4) if margin is None else margin,
            )

        return fig
    
    def make_color_map(self, parameter_values):
        # Normalize parameter values to range [0,1]
        normalized_values = (parameter_values - np.min(parameter_values)) / (np.max(parameter_values) - np.min(parameter_values))
        # Create color map using jet colormap
        color_map = cm.coolwarm(normalized_values)
        return color_map