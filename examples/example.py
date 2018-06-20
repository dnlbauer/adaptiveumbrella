#!/usr/bin/env python3
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

import sys

from adaptiveumbrella.wham2d import WHAM2DRunner

sys.path.append('..')





class MyUmbrellaRunner(WHAM2DRunner):

    def __init__(self):
        WHAM2DRunner.__init__(self)

    cum_frames = [0]

    def after_run_hook(self):
        filename = "tmp/pmf_{}.pdf".format("%02d" % self.num_iterations)
        print("Writing new pmf to {}".format(filename))
        pmf_to_plot = deepcopy(self.pmf)
        pmf_to_plot[pmf_to_plot < 0] = None
        pmf_to_plot[self.sample_list == 0] = None
        pmf_to_plot = pmf_to_plot.T

        self.cum_frames.append(len(self.sample_list[self.sample_list > 0]))

        fig, (ax0, ax1) = plt.subplots(ncols=2)
        im = ax0.imshow(pmf_to_plot, origin='lower', cmap='jet')
        cb = fig.colorbar(im, ax=ax0, orientation='horizontal', pad=0.15)
        cb.set_label("kJ/mol")

        ax1.plot(self.cum_frames, linewidth=0.5, marker="o", color='black')
        ax1.set_xlabel("Cycles")
        ax1.set_ylabel("Number of umbrella Windows")

        # ticks = [(x,x) for x in [-3, -2, -1, 0, 1, 2, 3]]
        # tick_positions = [ self._get_index_for_lambdas(x)[0] for x in ticks ]
        # tick_labels = [ str(x[0]) for x in ticks ]
        # ax0.set_yticks(tick_positions)
        # ax0.set_yticklabels(tick_labels)
        # ax0.set_xticks(tick_positions)
        # ax0.set_xticklabels(tick_labels)
        ax0.set_ylabel("$\phi$")
        ax0.set_xlabel("$\psi$")

        fig.subplots_adjust(wspace=.5)


        plt.savefig(filename, bbox_inches='tight', dpi=200)
        os.system("cp {} {}".format(filename, "tmp/pmf_current.pdf"))


    def simulate_frames(self, lambdas, frames):
        print("{} new simulations:".format(len(lambdas)))
        counter = 0

        if not os.path.exists("tmp"):
            os.mkdir('tmp')

        threads = []
        for f in lambdas:
            counter += 1
            print("{}) Simulate lambda1={}, lambda2={}".format(counter, *f))
            command = "bash data/sim.sh {} {} 2>&1 > tmp/run.log".format(*f)
            # print("Running {}".format(command))
            os.system(command)

runner = MyUmbrellaRunner()
runner.WHAM_EXEC = "/opt/wham/wham-2d/wham-2d"
runner.whamconfig = {
    'Px': 'pi',
    'hist_min_x': -3,
    'hist_max_x': 3,
    'num_bins_x': 100,
    'Py': 'pi',
    'hist_min_y': -3,
    'hist_max_y': 3,
    'num_bins_y': 100,
    'tolerance': 0.1,
    'fc_x': 100,
    'fc_y': 100
}


runner.cvs = np.array([
    (-3, 3, 0.2),
    (-3, 3, 0.2),
])
runner.cvs_init = (0, 0)
runner.E_min = 5
runner.E_max = 100
runner.E_incr = 10
runner.max_iterations = 5

runner.run()
