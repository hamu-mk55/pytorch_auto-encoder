import os
import sys
import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt


class _PlotData:
    def __init__(self, phase: str, yscale: str = 'linear'):
        self.phase = phase
        self.yscale = yscale

        self.epoch = []
        self.data = []

    def append(self, epoch, val):
        self.epoch.append(epoch)
        self.data.append(val)


class TrainLogger:
    def __init__(self):
        self.out_dir: str = None
        self.plot_dict: dict[str, list[_PlotData]] = {}
        self.metrics: float = None

        self.init()

    def init(self, out_dir='./log', id=None):
        # set out-dir
        if id is None:
            id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        self.out_dir = f'{out_dir}/{id}'
        os.makedirs(self.out_dir, exist_ok=True)

        # reset
        self.plot_dict = {}
        self.metrics = None

    def print_obj(self, obj, filename='out.txt'):
        sys.stdout = open(f'{self.out_dir}/{filename}', 'a')
        print(obj)
        sys.stdout = sys.__stdout__

    def save_dict(self, dict, filename='out.txt'):
        sys.stdout = open(f'{self.out_dir}/{filename}', 'a')
        for key, val in dict.items():
            print(f'{key}:   {val}')
        sys.stdout = sys.__stdout__

    def save_model(self, model, modelname, metrics=None):
        if metrics is None:
            torch.save(model.state_dict(), f'{self.out_dir}/{modelname}')
            return

        if self.metrics is None:
            self.metrics = metrics
        if metrics <= self.metrics:
            self.metrics = metrics
            torch.save(model.state_dict(), f'{self.out_dir}/{modelname}')

        return

    def init_plot(self, category, phase, yscale='linear'):
        plot = _PlotData(phase, yscale=yscale)

        if category not in list(self.plot_dict.keys()):
            self.plot_dict[category] = [plot]
        else:
            self.plot_dict[category].append(plot)

    def set_plot(self, category, phase, epoch, val):
        if val is None: return

        if category not in list(self.plot_dict.keys()):
            KeyError(f'ERROR in TrainLogger: NO KEY: {category}')

        plots = self.plot_dict[category]
        for plot in plots:
            if plot.phase == phase:
                plot.append(epoch, val)
                return

        KeyError(f'ERROR in TrainLogger: NO KEY: {category}/{phase}')

    def save_plot(self, category):
        if category not in list(self.plot_dict.keys()):
            KeyError(f'ERROR in TrainLogger: NO KEY: {category}')

        file_path = f'{self.out_dir}/{category}.png'

        # plot-----------
        plots = self.plot_dict[category]

        fig = plt.figure(0)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('epoch')
        ax.set_ylabel(category)
        ax.grid()

        colors = ['blue', 'black', 'red', 'green']
        _x_max = 10
        for cnt, plot in enumerate(plots):
            _color = colors[cnt % len(colors)]
            ax.plot(plot.epoch, plot.data,
                    color=_color, label=plot.phase)

            _x_max = max(_x_max, np.max(plot.epoch))

            _y_max = np.max(plot.data)
            if _y_max > 0:
                _y_min = 0
            else:
                _y_min = np.min(plot.data)
                _y_max = 0
            _yscale = plot.yscale

        ax.set_xlim([0, _x_max])
        if _yscale != 'log':
            ax.set_ylim([_y_min, _y_max])

        ax.set_yscale(_yscale)

        ax.legend(loc=0)
        fig.tight_layout()
        plt.savefig(file_path)
        fig.clf()


if __name__ == '__main__':
    exit()
