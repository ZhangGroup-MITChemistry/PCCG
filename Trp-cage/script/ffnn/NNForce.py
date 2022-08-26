import torch
torch.set_default_dtype(torch.float64)
import torch.nn as nn
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/MMFlow/")
from MMFlow import transform, MMFlow, utils
import openmm.unit as unit

class NNForceAngleDihedral(torch.nn.Module):
    def __init__(self, num_angles, num_dihedrals, hidden_size):
        super(NNForceAngleDihedral, self).__init__()

        self.num_angles = num_angles
        self.num_dihedrals = num_dihedrals
        self.hidden_size = hidden_size
        
        self.fc1 = torch.nn.Linear(num_angles + 2*num_dihedrals, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)        
        self.fc4 = torch.nn.Linear(hidden_size, 1, bias = False)        

    def forward(self, angles, dihedrals):
        x = torch.cat((torch.cos(angles), torch.cos(dihedrals), torch.sin(dihedrals)), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))        
        output = self.fc4(x)
        output = torch.squeeze(output)
        return output

class NNForceXYZ(torch.nn.Module):
    def __init__(self,
                 T,
                 hidden_size,
                 particle_index_for_angle,
                 particle_index_for_dihedral):        
        super(NNForceXYZ, self).__init__()

        self.T = T
        self.hidden_size = hidden_size
        self.particle_index_for_angle = torch.nn.Parameter(
            particle_index_for_angle,
            requires_grad = False)
        self.particle_index_for_dihedral = torch.nn.Parameter(
            particle_index_for_dihedral,
            requires_grad = False)

        self.nnforce_angle_dihedral = NNForceAngleDihedral(
            num_angles = self.particle_index_for_angle.shape[0],
            num_dihedrals = self.particle_index_for_dihedral.shape[0],
            hidden_size = self.hidden_size)

        self.Kb = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
        self.Kb = self.Kb.value_in_unit(unit.kilojoule_per_mole/unit.kelvin)
        
    def forward(self, positions):
        xyz = torch.unsqueeze(positions, 0)
        angles = utils.functional.compute_angles(xyz, self.particle_index_for_angle)        
        dihedrals = utils.functional.compute_dihedrals(xyz, self.particle_index_for_dihedral)
        potential_energy = self.Kb*self.T*self.nnforce_angle_dihedral(angles, dihedrals)
        return potential_energy    
