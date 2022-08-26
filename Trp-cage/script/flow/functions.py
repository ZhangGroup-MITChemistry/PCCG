import torch
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/MMFlow")
from MMFlow import utils
import mdtraj
import pickle
from sys import exit
import numpy as np

import simtk.openmm.app  as app
import simtk.openmm as omm
import simtk.unit as unit
import math

# ONE_4PI_EPS0 = 138.935456
# EPS0 = 1./(ONE_4PI_EPS0*4*math.pi)
# kbT = unit.BOLTZMANN_CONSTANT_kB*300*unit.kelvin*unit.AVOGADRO_CONSTANT_NA
# kbT = kbT.value_in_unit(unit.kilojoule_per_mole)
# ionic_strength = 2/(40*unit.angstroms)**3
# ionic_strength = ionic_strength.value_in_unit(unit.nanometer**(-3))
# lambda_D = math.sqrt(EPS0*kbT/(2*ionic_strength))

def make_custom_elec_force():
    formula = ["ONE_4PI_EPS0*charge1*charge2/(r*epsilon)*exp(-r/lambda_D)",
               "lambda_D = sqrt(EPS0*epsilon*kb*Temperature/(2*ionic_strength))",     
               "epsilon = A + B/(1 + kappa*exp(-lambda*B*r))",
               "EPS0 = 1./(ONE_4PI_EPS0*4*PI)",
               "PI = 3.141592653",
               "B = 78.4 - A; A = -8.5525; kappa = 7.7839; lambda = 0.03627",
               "kb = 2.4943387854459713/300",
               "ONE_4PI_EPS0 = 138.935456"
    ]
    
    custom_elec_force = omm.CustomNonbondedForce(";".join(formula))
    custom_elec_force.addGlobalParameter("Temperature", 300)
    ionic_strength = 150*unit.millimolar
    ionic_strength = ionic_strength*unit.AVOGADRO_CONSTANT_NA
    ionic_strength = ionic_strength.value_in_unit(unit.nanometer**(-3))
    custom_elec_force.addGlobalParameter("ionic_strength", ionic_strength)
    
    custom_elec_force.addPerParticleParameter('charge')

    print("remember to set values for global parameters T and ionic_strength")
    return custom_elec_force



def ic_to_feature_and_context(ic):
    feature = []
    context = []
    circular_feature_flag = []

    context = torch.cat((ic.reference_particle_2_bond[:, None],
                         ic.reference_particle_3_bond[:, None],
                         ic.bond), dim = -1)

    feature = torch.cat((ic.reference_particle_3_angle[:, None]/math.pi,
                         ic.angle/math.pi,
                         ic.dihedral), dim = -1)
    
    circular_feature_flag = torch.tensor(
        [False] +
        [False for i in range(ic.angle.shape[-1])] +
        [True for i in range(ic.dihedral.shape[-1])])
    
    logabsdet = torch.log(1./torch.tensor(math.pi)) * (1.0 + ic.angle.shape[-1])
    
    return context, feature, circular_feature_flag, logabsdet

def feature_and_context_to_ic(context, feature):
    ic = {}
    
    ic['reference_particle_1_xyz'] = feature.new_zeros(feature.shape[0], 3)
    ic['reference_particle_2_bond'] = context[:, 0]
    ic['reference_particle_3_bond'] = context[:, 1]
    ic['reference_particle_3_angle'] = feature[:, 0] * math.pi
    ic['bond'] = context[:, 2:]
    ic['angle'] = feature[:, 1:1+ic['bond'].shape[-1]] * math.pi
    ic['dihedral'] = feature[:, 1+ic['bond'].shape[-1]:]

    logabsdet = torch.log(torch.tensor(math.pi)) * (1.0 + ic['angle'].shape[-1])

    ic = utils.InternalCoordinate(
        ic['reference_particle_1_xyz'],
        ic['reference_particle_2_bond'],
        ic['reference_particle_3_bond'],
        ic['reference_particle_3_angle'],
        ic['bond'], ic['angle'], ic['dihedral']
    )

    return ic, logabsdet
