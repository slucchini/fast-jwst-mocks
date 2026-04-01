import numpy as np, arepo
from snaptools import sim_utils as utils
from scipy.spatial import KDTree
import astropy.units as u

###
### Load the snapshot with Inspector Gadget <https://inspector-gadget.readthedocs.io/en/latest/>
###
outdir = "./"
s = arepo.Snapshot(outdir+"snapshot.hdf5")
# Recenter
for g in s.groups:
    g.pos -= s.BoxSize/2.

###
### Save the snapshot data in txt files accessible by SKIRT
###

header = """# dust_data.txt: import file for Voronoi mesh media -- dust
# Column 1: position x (pc)
# Column 2: position y (pc)
# Column 3: position z (pc)
# Column 4: gas mass (Msun)
# Column 5: metallicity (1)
# Column 6: temperature (K)
# Column 7: velocity vx (km/s)
# Column 8: velocity vy (km/s)
# Column 9: velocity vz (km/s)"""

dust_mass = s.part0.mass.value*1e10

data = np.hstack((s.part0.pos.value*1e3,np.array([dust_mass]).T,
                  np.array([s.part0.gz]).T,np.array([s.part0.temp]).T,
                  s.part0.vel.value))

np.savetxt(outdir+"dust_data.txt",data,header=header,fmt="%.9e")



tree = KDTree(s.part4.pos)
hsml = tree.query(s.part4.pos,k=[32])[0]
hsml_max = 0.3
hsml[hsml > hsml_max] = hsml_max

header = """# star_data.txt: import file for particle source
# Column 1: position x (pc)
# Column 2: position y (pc)
# Column 3: position z (pc)
# Column 4: smoothing length (pc)
# Column 5: velocity vx (km/s)
# Column 6: velocity vy (km/s)
# Column 7: velocity vz (km/s)
# Column 8: mass (Msun)
# Column 9: metallicity (1)
# Column 10: age (yr)"""

stellar_ages = (np.array(s.time - s.part4.gage)*u.kpc/(u.km/u.s)).to(u.yr).value
is_HII = stellar_ages < 10e6

star_data = np.hstack((s.part4.pos[~is_HII]*1e3,hsml[~is_HII]*1e3,s.part4.vel[~is_HII],
                       np.array([s.part4.mass[~is_HII]]).T*1e10,np.array([s.part4.gz[~is_HII]]).T,
                       np.array([stellar_ages[~is_HII]]).T))

np.savetxt(outdir+"star_data.txt",star_data,header=header,fmt="%.9e")



header = """# HII_data.txt: import file for particle source
# Column 1: position x (pc)
# Column 2: position y (pc)
# Column 3: position z (pc)
# Column 4: size h (pc)
# Column 5: velocity x (km/s)
# Column 6: velocity y (km/s)
# Column 7: velocity z (km/s)
# Column 8: star formation rate (Msun/yr)
# Column 9: metallicity (1)
# Column 10: compactness (1)
# Column 11: pressure (Pa)
# Column 12: covering factor (1)"""

# from https://ui.adsabs.harvard.edu/abs/2024A%26A...683A.181B/abstract

cmask = is_HII

sfr = s.part4.gima*1e10/1e7
rng = np.random.default_rng()
compactness = rng.normal(5,0.4,size=np.sum(cmask)) # from https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.5703K/abstract
press = np.ones(np.sum(cmask))*1.38e-12
cf = np.exp(-stellar_ages[cmask]/3e6)

star_data = np.hstack((s.part4.pos[cmask]*1e3,hsml[cmask]*1e3,s.part4.vel[cmask],
                       np.array([sfr[cmask]]).T,np.array([s.part4.gz[cmask]]).T,np.array([compactness]).T,
                       np.array([press]).T,np.array([cf]).T))

np.savetxt(outdir+"HII_data.txt",star_data,header=header,fmt="%.9e")