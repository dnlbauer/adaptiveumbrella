import os
import subprocess
import pandas as pd
import numpy as np
import math

# ignore SettingsWithCopyWarning
pd.options.mode.chained_assignment = None

from adaptiveumbrella import UmbrellaRunner


class WHAM2DRunner(UmbrellaRunner):
    """ Umbrella runner implementation that uses wham-2d to perform
    the pmf calculation.

    Attributes:
        WHAM_EXEC: path to wham executeable

        """

    def __init__(self):
        UmbrellaRunner.__init__(self)
        self.WHAM_EXEC = 'wham-2d'
        self.tmp_folder = "tmp/WHAM"
        self.simulation_folder = "tmp/simulations"

        if not os.path.exists(self.tmp_folder):
            os.makedirs(self.tmp_folder)

    def create_metadata_file(self):
        """ create the metadata file for wham-2d """
        path = os.path.join(self.tmp_folder, "{}_metadata.dat".format(self.num_iterations))
        with open(path, 'w') as out:
            for file in os.listdir(self.simulation_folder):
                if not os.path.exists(os.path.join(path, file, "COLVAR")):
                    continue
                prefix, x, y = file.split("_")
                colvar_file = os.path.join(file, 'COLVAR')
                out.write("{file}\t{x}\t{y}\t{fc_x}\t{fc_y}\n".format(
                    file=os.path.join(self.simulation_folder, colvar_file), x=x, y=y, fc_x=self.whamconfig['fc_x'], fc_y=self.whamconfig['fc_y']
                ))
        return path


    def get_wham_output_file(self):
        """ Output file for wham-2d """
        return os.path.join(self.tmp_folder, 'freeenergy_tmp.dat')

    def run_wham2d(self, metafile_path, output_path):
        """ Runs wham-2d with the given parameters. See http://membrane.urmc.rochester.edu/sites/default/files/wham/doc.html """

        cmd = "{exec} Px={px} {min_x} {max_x} {frames_x} Py={py} {min_y} {max_y} {frames_y} {tol} 298 0 {metafile} {outfile} 0".format(
            exec=self.WHAM_EXEC,
            px=self.whamconfig['Px'],
            min_x=self.whamconfig['hist_min_x'],
            max_x=self.whamconfig['hist_max_x'],
            frames_x=self.whamconfig['num_bins_x'],
            py=self.whamconfig['Py'],
            min_y=self.whamconfig['hist_min_y'],
            max_y=self.whamconfig['hist_max_y'],
            frames_y=self.whamconfig['num_bins_y'],
            tol=self.whamconfig['tolerance'],
            metafile=metafile_path,
            outfile=output_path
        )
        print(cmd)
        err_code = subprocess.call(cmd, shell=True)
        if err_code != 0:
            print("wham exited with error code {}".format(err_corde))
            exit(1)

    def load_wham_pmf(self, wham_file):
        """ Load the new pmf into a pandas dataframe """
        df = pd.read_csv(wham_file, delim_whitespace=True, names=['x', 'y', 'e', 'pro'], skiprows=1, index_col=None)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['e'], how='all')
        return df

    def update_pmf(self, wham_pmf):
        """ Update the internal pmf representation from the pmf generated with wham """
        # map all points of self.pmf to one point of the wham_pmf and update self.pmf accordingly
        for x in range(self.pmf.shape[0]):
            for y in range(self.pmf.shape[1]):
                lambdax, lambday = self._get_lambdas_for_index((x, y))
                reduced = wham_pmf[ (abs(wham_pmf.x-lambdax) < self.cvs[0][2]) & (abs(wham_pmf.y-lambday) < self.cvs[1][2]) ]
                if len(reduced) == 0:
                    continue

                reduced['dist'] = reduced.apply(lambda row: np.linalg.norm((lambdax-row['x'], lambday-row['y'])), axis=1)
                min = reduced[reduced.dist == reduced.dist.min()]
                min_row = reduced[(reduced.x == min.x.iloc[0]) & (reduced.y == min.y.iloc[0])]
                self.pmf[x, y] = min_row.e.iloc[0]



    def calculate_new_pmf(self):
        print("Running wham-2d")
        metafile_path = self.create_metadata_file()
        wham_pmf_file = self.get_wham_output_file()
        self.run_wham2d(metafile_path, wham_pmf_file)

        wham_pmf = self.load_wham_pmf(wham_pmf_file)
        self.update_pmf(wham_pmf)
        return self.pmf
