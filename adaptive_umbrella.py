#!/usr/bin/env python3
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod


#%%
class UmbrellaRunner(ABC):
    def init_pmf(self):
        """ returns a NxM matrix filled with -1 where N and M are the total number
        of lambda frames along the reaction coordinates as determined by
        self.lambda_min, self.lambda_max and self.lambda_delta """
        ranges = []
        for dimen in range(len(self.lambda_delta)):
            ranges.append(np.arange(self.lambda_min[dimen], self.lambda_max[dimen]+self.lambda_delta[dimen], self.lambda_delta[dimen]))

        mesh = np.meshgrid(*ranges)
        pmf = np.zeros(np.dstack(mesh).shape[:-1]) # this is magic
        pmf[:] = -1
        return pmf
        
    def get_lambdas_for_index(self, idx):
        """ takes a coordinate tuple and returns corresponding lambda values """
        lambdas = self.lambda_min + idx*self.lambda_delta
        return np.round(lambdas, 10)

    def get_index_for_lambdas(self, lambdas):
        """ takes a lambda tuple and returns corresponding indeces of the pmf instance variable"""
        idx = []
        for dimen in range(len(lambdas)):
            r = np.arange(self.lambda_min[dimen], self.lambda_max[dimen]+self.lambda_delta[dimen], self.lambda_delta[dimen])
        
            for i in range(len(r)):
                if abs(r[i]-lambdas[dimen]) < 0.00001:
                    idx.append(i)
                    break
        return idx

    # TODO make this work with more then 2 dimensions
    def get_root_frames(self, pmf, E_max):
        """ returns the index of all positions in the pmf where the energy is
        smaller W_max and greater 0 """
        selection = np.where((pmf <= E_max) & (pmf >= 0))
        frames = []
        for i in range(len(selection[0])):
            frames.append((selection[0][i], selection[1][i]))
        return frames

    # TODO make this work with more then 2 dimensions
    def get_new_frames(self, pmf, root_frames):
        """ returns a dict of all frames surrounding the root_frames
        that have not an assigned energy yet, as well as their corresponding root
        frame in the format {new_frame1: root_frame1, new_frame2: root_frame2} """
        
        # find all neighboring frames and create a dict that associates them to their
        # root frames
        new_frames = {}
        for frame in root_frames:
            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    new_frame = list(frame)
                    new_frame[0] += x
                    new_frame[1] += y
                    if new_frame[0] < 0 or new_frame[0]+1 > len(self.pmf[0]) or new_frame[1] < 0 or new_frame[1]+1 > len(self.pmf[1]):
                        continue
                    try:
                        old_root = new_frames[tuple(new_frame)]
                        if old_root is not None:    
                            old_start_energy = pmf[old_root[0], old_root[1]]
                            new_start_energy = pmf[frame[0], frame[1]]
                            if old_start_energy > new_start_energy:
                                new_frames[tuple(new_frame)] = frame
                    except KeyError:
                        new_frames[tuple(new_frame)] = frame
    
        # remove already sampled frames (energy >= 0)
        new_frames_list = list(new_frames.keys())
        for idx in range(len(new_frames_list)):
            new_frame = new_frames_list[idx]
            energy = pmf[new_frame]
            if energy >= 0:
                del(new_frames[new_frame])
                
        return new_frames
    
    
    def main(self):
        # get the initial simulation and surrounding frames
        root_frames = [self.get_index_for_lambdas(self.lambda_init)]
        new_frames = self.get_new_frames(self.pmf, root_frames)
        
        self.num_iterations = 0
        
        # outer main loop: increase E and calculate PMF until E > E_max
        while True:
            
            # stop if max iterations is reached
            self.num_iterations += 1
            if(self.max_iterations > 0 and self.num_iterations > self.max_iterations):
                print("Max iterations reached ({})".format(self.max_iterations))
                return
            
            self.E = self.E_min
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Iteration: {} (max={})".format(self.num_iterations, self.max_iterations))
            new_lambdas = [ self.get_lambdas_for_index(x) for x in new_frames.keys() ]
            
            print("Running simulations")
            self.simulate_frames(new_frames, new_lambdas)
            
            print("Calculating new PMF")
            self.pmf = self.wham()
            self.after_run_hook()
            
            while self.E <= self.E_max:
                root_frames = self.get_root_frames(self.pmf, self.E)
                new_frames = self.get_new_frames(self.pmf, root_frames)
                
                if len(new_frames) == 0:
                    self.E += self.E_incr
                    print("Max energy increased to {} (max={})".format(self.E, self.E_max))
                else:
                    break
            

    def run(self):
        self.pmf = self.init_pmf()
        self.main()
        print("Umbrella sampling finished.")
        
    @abstractmethod
    def simulate_frames(new_frames, new_lambdas):
        pass
    
    @abstractmethod
    def wham():
        pass
    
    def after_run_hook(self):
        pass
                
if __name__ == "__main__":
    from copy import deepcopy
    import subprocess
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    class MyUmbrellaRunner(UmbrellaRunner):
        
        def wham(self):
            # copy colvar files
            os.system("mkdir -p WHAM")
            for f in os.listdir("sim"):
                with open("sim/{}/COLVAR".format(f), "r") as i:
                    with open("WHAM/{}.xvg".format(f), 'w') as o:
                        for line in i.readlines()[100:]:
                            o.write(line)

                    
            # generate metadata.dat
            x_vals = []
            y_vals = []
            metadata_file = "WHAM/{}_metadata.dat".format(self.num_iterations)
            with open(metadata_file, 'w') as o:
                for f in os.listdir("sim"):
                    prefix, x, y = f.split("_")
                    x_vals.append(float(x))
                    y_vals.append(float(y))
                    o.write("WHAM/{}.xvg {} {} {} {}\n".format(f, x,y, 100, 100))
            x_vals = np.array(list(set(x_vals)))
            y_vals = np.array(list(set(y_vals)))

            # min_x = x_vals.min() - self.lambda_delta[0]
            # min_y = y_vals.min() - self.lambda_delta[1]
            # max_x = x_vals.max() + self.lambda_delta[0]
            # max_y = y_vals.max() + self.lambda_delta[1]
            min_x, min_y = self.lambda_min
            max_x, max_y = self.lambda_max
            frames_x, frames_y = 1002, 1002
            print("Running WHAM-2d:")
            wham_output = "WHAM/{}_freeenergy.dat".format(self.num_iterations)
            cmd = "/opt/wham/wham-2d/wham-2d Px=pi {min_x} {max_x} {frames_x} Py=pi {min_y} {max_y} {frames_y} 0.1 298 0 {metafile} {outfile} 0".format(
                      	min_x=min_x,
                      	max_x=max_x,
                      	frames_x=frames_x,
                      	min_y=min_y,
                      	max_y=max_y,
                      	frames_y=frames_y,
                      	metafile=metadata_file,
                      	outfile=wham_output
                      	)
            print(cmd)
            os.system(cmd)
        
            print("Update pmf from wham")
            df = pd.read_csv(wham_output, delim_whitespace=True, names=['x','y','e', 'pro'], skiprows=1, index_col=None)
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['e'], how='all')
            new_pmf = deepcopy(self.pmf)
            for x in range(new_pmf.shape[0]):
                for y in range(new_pmf.shape[1]):
                    lambdax, lambday = self.get_lambdas_for_index((x,y))
                    x_selection = (df.x-lambdax).abs() < 0.01
                    y_selection = (df.y-lambday).abs() < 0.01
                    selected_energies = df[(x_selection) & (y_selection)].e
                    if len(selected_energies) == 0:
                        new_pmf[x,y] = -1
                    else:
                        new_pmf[x,y] = selected_energies.iloc[0] 
                        
                    
            return new_pmf
            
        def simulate_frames(self, new_frames, new_lambdas):
            print("{} new simulations:".format(len(new_lambdas)))
            counter = 0

            threads = []
            for f in new_lambdas:
                counter += 1
                if os.path.exists("sim/sim_{}_{}/COLVAR".format(*f)):
                    print("{}) Skipping lambdas={}/{}: COLVAR exists".format(counter, *f))
                    continue
                
                print("{}) Simulate lambda1={}, lambda2={}".format(counter, *f))
                command = "bash sim.sh {} {} 2>&1 > run.log".format(*f)
                # print("Running {}".format(command))
                os.system(command)

        def after_run_hook(self):
            filename = "pmf_{}.pdf".format(self.num_iterations)
            print("Writing new pmf to {}".format(filename))
            pmf_to_plot = deepcopy(self.pmf.T)
            pmf_to_plot[pmf_to_plot < 0] = None
            plt.figure()
            plt.imshow(pmf_to_plot, origin="lower", cmap='jet')
            cb = plt.colorbar(pad=0.1)
            cb.set_label("kJ/mol")
            plt.savefig(filename)
            os.system("cp {} {}".format(filename, "pmf_current.pdf"))
                
                
    
    runner = MyUmbrellaRunner()
    runner.lambda_max = np.array((3.1, 3.1))
    runner.lambda_min = -runner.lambda_max
    runner.lambda_delta = np.array((0.1, 0.1))
    runner.lambda_init = np.array((1,-1.4))
    runner.E_min = 5
    runner.E_max = 100
    runner.E_incr = 10
    runner.max_iterations = 10
    
    runner.run()




