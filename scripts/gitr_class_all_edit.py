#python library tools for gitr

import sys,os
import numpy as np
import io,libconf
import math
from numpy import linalg as LA

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LogNorm

# imports within other functions
# from netCDF4 import Dataset

_VERSION = 1.0
_LAST_UPDATE = 'Aug. 10. 2018'

# ----------------------------------------------------------------------------------------
# general figure font configuration, global change throughout python session
FONTSIZE = 18
if 'Anaconda' in sys.version:
	plt.rcParams['font.sans-serif'] = 'Arial'	# anaconda
	plt.rcParams['font.serif'] = 'Times'	# anaconda
else:
	plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Nimbus Sans L', 'Liberation Sans', 'DejaVu Sans']
	plt.rcParams['font.serif'] = ['Times', 'Times New Roman', 'Nimbus Roman No9 L', 'DejaVu Serif']
FONT = {'family' : 'sans-serif',
		'weight' : 'normal', # normal, bold
		'size'   : FONTSIZE} 
plt.rc('font', **FONT)

# ----------------------------------------------------------------------------------------
# --- gitr Class -------------------------------------------------------------------------

class gitr:
	"""
	Load, Post-process and Plot GITR output
	"""

	def __init__(self, path = './', impurity = 'W', show = False):
		"""
		path (string)      Path to top folder with input and output folders, default './'
		impurity (string)  Main impurity: 'W' (default), ???
		show (bool)        True: make all available plots
		"""
		if not (path[-1] == '/'): path += '/'
		self.path = path
		
		self.config = {}
		# load input configurations
		self.InputFiles = [] #['gitrInput','gitrGeometry','particleSource']
		for key in self.InputFiles:
			file = self.path + 'input/' + key + '.cfg'
			if os.path.isfile(file): 
				with io.open(file) as f: self.config[key] = libconf.load(f)
		
		self.data = {}
		# load NetCDF output files
		self.files = ['surface','spec', 'compare-spec', 'compare-surface'] #['surface','history','positions','particleSource','forces', 'spec']
		for key in self.files:
			file = self.path + 'output/' + key + '.nc'
			if os.path.isfile(file): self.data[key] = openNCfile(file)
		
		# set impurity
		self.impurity = impurity
		if self.impurity in ['W','w','tungsten','Tungsten']:
			self.Z = 74
			self.M = 184
		
		self.Pmass = 1.66e-27
		self.eCharge = 1.602e-19
		
		# filter NaN
		#if self.data.has_key('positions'): 
		if 'positions' in self.data:
			self.NotNan = (~np.isnan(self.data['positions']['x'])) & (~np.isnan(self.data['positions']['y'])) & (~np.isnan(self.data['positions']['z']))
		
		if show: self.plotAll()
		
	
	def plotAll(self, N = None):
		"""
		Make all plots 
		N (int)  number of orbits to plot in 3D, default: all
		"""
		#self.plot2dGeom()  
		#self.plot3dGeom()
		#self.plotParticles()
		#self.plotParticleHistograms()
		#self.plotOrbits(N)
		self.plotDensity()
		#self.plotErosion()
                self.plotEnergyDist()
                #self.plotEnergyDistLog()
		self.plotReflDist()
		self.plotSputtDist()
	  
	def plotParticles(self):
		"""
		Plot final particle positions
		"""
		#if not self.data.has_key('positions'): 
		if 'positions' not in self.data:
			print( 'positions.nc not available')
			return
			
		x = self.data['positions']['x']
		y = self.data['positions']['y']
		z = self.data['positions']['z']
		hit = self.data['positions']['hitWall'] > 0
		notHit = self.data['positions']['hitWall'] == 0
		r = np.sqrt(x**2 + y**2)
				
		fig = plt.figure(figsize = (9,6))
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(x[notHit],y[notHit],z[notHit], c = 'b')
		ax.scatter(x[hit],y[hit],z[hit], c = 'r')
		plt.title('Final ' + self.impurity + ' Particle Positions', fontsize = FONTSIZE)
		ax.view_init(azim = 225)
		ax.ticklabel_format(style = 'sci',useOffset = True, useMathText = True, scilimits = (2,2))
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')

		plt.figure(figsize = (9,6))
		plt.scatter(r[notHit],z[notHit], c = 'b')
		plt.scatter(r[hit],z[hit], c = 'r')
		plt.title('Final ' + self.impurity + ' Particle Positions', fontsize = FONTSIZE)
		plt.xlabel('r [m]')
		plt.ylabel('z [m]')


	def plotParticleHistograms(self):
		"""
		Plot particle, energy and charge deposition patterns on the target surface
		"""
		#if not self.data.has_key('positions'): 
		if 'positions' not in self.data:
			print( 'positions.nc not available')
			return

		x = self.data['positions']['x']
		y = self.data['positions']['y']
		z = self.data['positions']['z']
		hit = (self.data['positions']['hitWall'] > 0) & self.NotNan
		
		fig = plt.figure(figsize = (10,6))
		Nx,Ny = 30,30
		H,yedges,xedges = np.histogram2d(y[hit], x[hit], bins = [Ny,Nx])   # vertical, horizontal
		X,Y = np.meshgrid(xedges, yedges)
		cs = plt.imshow(np.log10(H), extent = [X.min(), X.max(), Y.min(), Y.max()], cmap = 'cool', origin = 'lower', aspect = 'auto', interpolation = 'nearest')
		#cs = plt.pcolormesh(X, Y, H, cmap = 'plasma')
		plt.xlabel('x [m]')
		plt.ylabel('y [m]')
		C = plt.colorbar(cs, pad = 0.01, format = '%.3g')
		C.set_label('Number of Particles', rotation = 270, size = FONTSIZE, va = 'bottom')
		C.ax.set_yticklabels(['10$^{' + item.get_text() + '}$' for item in C.ax.get_yticklabels()])
		plt.title('Target Erosion: Particles', fontsize = FONTSIZE)


		xs = x[hit]
		ys = y[hit]
		energies = np.zeros((Ny,Nx))
		charges = np.zeros((Ny,Nx))

		vx = self.data['positions']['vx']
		vy = self.data['positions']['vy']
		vz = self.data['positions']['vz']
		E = 0.5*self.M*self.Pmass* (vx**2 + vy**2 + vz**2) / self.eCharge
		Ehit = E[hit]
		chargeHit = self.data['positions']['charge'][hit]
		
		for j in xrange(Ny):
			if j == Ny-1: fy = (ys >= yedges[j]) & (ys <= yedges[j+1])
			else: fy = (ys >= yedges[j]) & (ys < yedges[j+1])
			for i in xrange(Nx):
				if i == Nx-1: fx = (xs >= xedges[i]) & (xs <= xedges[i+1])
				else: fx = (xs >= xedges[i]) & (xs < xedges[i+1])
				fs = fx & fy
				energies[j,i] = Ehit[fs].mean()          # Note: mean of an empty array is nan
				charges[j,i] = chargeHit[fs].mean()

		fig = plt.figure(figsize = (10,6))		
		cs = plt.imshow(energies, extent = [X.min(), X.max(), Y.min(), Y.max()], cmap = 'cool', origin = 'lower', aspect = 'auto', interpolation = 'nearest')
		plt.xlabel('x [m]')
		plt.ylabel('y [m]')
		C = plt.colorbar(cs, pad = 0.01, format = '%.3g')
		C.set_label('Particle Energy [eV]', rotation = 270, size = FONTSIZE, va = 'bottom')
		plt.title('Target Erosion: Energy', fontsize = FONTSIZE)

		fig = plt.figure(figsize = (10,6))		
		cs = plt.imshow(charges, extent = [X.min(), X.max(), Y.min(), Y.max()], cmap = 'cool', origin = 'lower', aspect = 'auto', interpolation = 'nearest')
		plt.xlabel('x [m]')
		plt.ylabel('y [m]')
		C = plt.colorbar(cs, pad = 0.01, format = '%.3g')
		C.set_label('Particle Charge [e]', rotation = 270, size = FONTSIZE, va = 'bottom')
		plt.title('Target Erosion: Charge', fontsize = FONTSIZE)


	def plotOrbits(self, N = None):
		"""
		Plot the 3D orbits of the first N particles (default is all)
		"""
		#if not self.data.has_key('history'):
		if 'history' not in self.data:
			print( 'history.nc not available')
			return

		x = self.data['history']['x']
		y = self.data['history']['y']
		z = self.data['history']['z']
		r = np.sqrt(x**2 + y**2)
		
		fig = plt.figure(figsize = (9,6))
		ax = fig.add_subplot(111, projection='3d')
		if N is None: N = self.data['history']['nP']
		for i in xrange(N):
			ax.plot(x[i,:],y[i,:],z[i,:])
		plt.title(self.impurity + ' Particle Orbits', fontsize = FONTSIZE)
		ax.view_init(azim = 225)
		ax.ticklabel_format(style = 'sci',useOffset = True, useMathText = True, scilimits = (2,2))
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		
		plt.figure(figsize = (9,6))
		for i in xrange(N):
			plt.plot(r[i,:],z[i,:])
		plt.title(self.impurity + ' Particle Orbits', fontsize = FONTSIZE)
		plt.xlabel('r [m]')
		plt.ylabel('z [m]')


	def plotDensity(self):
		"""
		Plot the particle density in the X-section
		"""
		#if not self.data.has_key('spec'): 
		if 'spec' not in self.data:
			print( 'spec.nc not available')
			return

		dens =  self.data['spec']['n'][-1,:,:,:]
		gridR =  self.data['spec']['gridR']
		gridY =  self.data['spec']['gridY']
		gridZ =  self.data['spec']['gridZ']
		sumdens = np.sum(dens,1)/len(self.data['spec']['gridY'])
		digit = int(np.floor(np.log10(sumdens.max())))
		if abs(digit) > 1:
			factor = 10**digit
			sumdens /= factor
		if 'compare-spec' in self.data:
                    copy = sumdens.copy()
                    copy[copy <= 0] = 0
		    mdens =  self.data['compare-spec']['n'][-1,:,:,:]
		    mgridR =  self.data['compare-spec']['gridR']
		    mgridY =  self.data['compare-spec']['gridY']
		    mgridZ =  self.data['compare-spec']['gridZ']
		    msumdens = np.sum(mdens,1)/len(self.data['compare-spec']['gridY'])
		    mdigit = int(np.floor(np.log10(msumdens.max())))
		    if abs(mdigit) > 1:
			    mfactor = 10**mdigit
			    msumdens /= mfactor
                    mcopy = msumdens.copy()
                    mcopy[mcopy <= 0] = 0
                    diffspec= copy - mcopy
                    norm0 = LA.norm(diffspec) #diffspec.ravel(), 0)
                    norm = LA.norm(sumdens)
                    print('density diff. norm',norm0, 'GITR',norm,'percent',norm0/norm*100)

                sumdens[sumdens <= 0] = np.nan		# makes areas with zero density white in the plot
                #np.set_printoptions(threshold=np.inf)
		fig = plt.figure(figsize = (10,6))		
		cs = plt.imshow(sumdens, extent = [gridR.min(), gridR.max(), gridZ.min(), gridZ.max()], cmap = 'cool', origin = 'lower', aspect = 'auto', interpolation = 'nearest')
		plt.xlabel('x [m]')
		plt.ylabel('z [m]')
		C = plt.colorbar(cs, pad = 0.01, format = '%.3g')
		if abs(digit) > 1:
			C.set_label('Density [count x 10$^{' + str(digit) + '}$]', rotation = 270, size = FONTSIZE, va = 'bottom')
		else:
			C.set_label('Density [count]', rotation = 270, size = FONTSIZE, va = 'bottom')
		plt.title('X-section Particle Density', fontsize = FONTSIZE)
		plt.plot(gridR, gridR* np.sin(np.pi/6.0), 'k-', lw = 2)

	def plotEnergyDistLog(self):
		"""
		Plot the Energy-Angle distribution of the particles
		"""
		#if not self.data.has_key('surface'): 
		if 'surface' not in self.data:
			print( 'surface.nc not available')
			return
		print( 'ploting surface.nc')
		edist = self.data['surface']['surfEDist'].flatten()
		edist = edist.reshape(self.data['surface']['nSurfaces'],self.data['surface']['nEnergies'],self.data['surface']['nAngles'])
		eDtotal = np.sum(edist,0).T
		#eDtotal *= 100.0/eDtotal.max()
		eDtotal[eDtotal <= 0] = np.nan		# makes areas with zero density white in the plot
		#fig = plt.figure(figsize = (10,6))	
		cs = plt.imshow(np.log10(eDtotal), extent = [0, 1000, 0, 90], cmap = 'cool', origin = 'lower', aspect = 'auto', interpolation = 'nearest')
		plt.xlabel('Energy [eV]')
		plt.ylabel('Angle [deg]')
		plt.xscale('log')
		C = plt.colorbar(cs, pad = 0.01, format = '%.3g')
		C.set_label('Particles [a.u.]', rotation = 270, size = FONTSIZE, va = 'bottom')
		plt.title('Energy Distribution', fontsize = FONTSIZE)

	def plotEnergyDist(self, all = False):
		"""
		Plot the Energy-Angle distribution of the particles
		"""
		#if not self.data.has_key('surface'): 
		if 'surface' not in self.data:
			print( 'surface.nc not available')
			return
		print( 'ploting surface.nc')
		if all:
			for key in self.data['surface'].keys():
				if key in ['nSurfaces','nAngles','nEnergies','surfEDist','surfaceNumber']: continue
				plt.figure(figsize = (9,6))
				plt.plot(self.data['surface']['surfaceNumber'],self.data['surface'][key], 'k-', lw = 2)
				plt.title(key, fontsize = FONTSIZE)
				plt.xlabel('Surface #')
				plt.ylabel('Value [a.u.]')
		edist = self.data['surface']['surfEDist'].flatten()
		edist = edist.reshape(self.data['surface']['nSurfaces'],self.data['surface']['nEnergies'],self.data['surface']['nAngles'])
		eDtotal = np.sum(edist,0).T
		#eDtotal *= 100.0/eDtotal.max()
                if 'compare-surface' in self.data:
                    copy = eDtotal.copy()
                    copy[copy <= 0] = 0
		    medist = self.data['compare-surface']['surfEDist'].flatten()
		    medist = medist.reshape(self.data['compare-surface']['nSurfaces'],self.data['compare-surface']['nEnergies'],self.data['compare-surface']['nAngles'])
		    meDtotal = np.sum(medist,0).T
                    mcopy = meDtotal.copy()
                    mcopy[mcopy <= 0] = 0
                    diffdist = (mcopy - copy)
                    #diffSelect = diffdist[diffdist>5]/copy[diffdist>5] * 100.0
                    norm0 = LA.norm(diffdist) #diffdist.ravel(), 0)
                    norm = LA.norm(eDtotal)
                    print('Energy-dist diff. norm', norm0,'GITR',norm,'percent',norm0/norm*100)
		eDtotal[eDtotal <= 0] = np.nan		# makes areas with zero density white in the plot
		fig = plt.figure(figsize = (10,6))		
		cs = plt.imshow(eDtotal, extent = [0, 1000, 0, 90], cmap = 'cool', origin = 'lower', aspect = 'auto', interpolation = 'nearest')
		plt.xlabel('Energy [eV]')
		plt.ylabel('Angle [deg]')
                #plt.xscale('log')
		C = plt.colorbar(cs, pad = 0.01, format = '%.3g')
		C.set_label('Particles [a.u.]', rotation = 270, size = FONTSIZE, va = 'bottom')
		plt.title('Energy Distribution', fontsize = FONTSIZE)
		
	def plotReflDist(self, all = False):
		"""
		Plot the Energy-Angle distribution of the particles
		"""
		#if not self.data.has_key('surface'): 
		if 'surface' not in self.data:
			print( 'surface.nc not available')
			return
		print( 'ploting surface.nc')
		if all:
			for key in self.data['surface'].keys():
				if key in ['nSurfaces','nAngles','nEnergies','surfReflDist','surfaceNumber']: continue
				plt.figure(figsize = (9,6))
				plt.plot(self.data['surface']['surfaceNumber'],self.data['surface'][key], 'k-', lw = 2)
				plt.title(key, fontsize = FONTSIZE)
				plt.xlabel('Surface #')
				plt.ylabel('Value [a.u.]')
		edist = self.data['surface']['surfReflDist'].flatten()
		edist = edist.reshape(self.data['surface']['nSurfaces'],self.data['surface']['nEnergies'],self.data['surface']['nAngles'])
		eDtotal = np.sum(edist,0).T
		#eDtotal *= 100.0/eDtotal.max()
                if 'compare-surface' in self.data:
                    copy = eDtotal.copy()
                    copy[copy <= 0] = 0
                    medist = self.data['compare-surface']['surfReflDist'].flatten()
                    medist = medist.reshape(self.data['compare-surface']['nSurfaces'],self.data['compare-surface']['nEnergies'],self.data['compare-surface']['nAngles'])
                    meDtotal = np.sum(medist,0).T
                    diffdist = meDtotal - eDtotal
                    norm0 = LA.norm(diffdist)
                    norm = LA.norm(eDtotal)
                    #print(diffdist[diffdist>5])
                    #print(eDtotal[diffdist>5])
                    print('Reflection diff. norm',norm0, 'GITR',norm,'percent',norm0/norm*100)
		eDtotal[eDtotal <= 0] = np.nan		# makes areas with zero density white in the plot
		fig = plt.figure(figsize = (10,6))		
		cs = plt.imshow(eDtotal, extent = [0, 1000, 0, 90], cmap = 'cool', origin = 'lower', aspect = 'auto', interpolation = 'nearest')
		plt.xlabel('Energy [eV]')
		plt.ylabel('Angle [deg]')
		C = plt.colorbar(cs, pad = 0.01, format = '%.3g')
		C.set_label('Particles [a.u.]', rotation = 270, size = FONTSIZE, va = 'bottom')
		plt.title('Reflection Distribution', fontsize = FONTSIZE)


	def plotSputtDist(self, all = False):
		"""
		Plot the Energy-Angle distribution of the particles
		"""
		#if not self.data.has_key('surface'): 
		if 'surface' not in self.data:
			print( 'surface.nc not available')
			return
		print( 'ploting surface.nc')
		if all:
			for key in self.data['surface'].keys():
				if key in ['nSurfaces','nAngles','nEnergies','surfSputtDist','surfaceNumber']: continue
				plt.figure(figsize = (9,6))
				plt.plot(self.data['surface']['surfaceNumber'],self.data['surface'][key], 'k-', lw = 2)
				plt.title(key, fontsize = FONTSIZE)
				plt.xlabel('Surface #')
				plt.ylabel('Value [a.u.]')
		edist = self.data['surface']['surfSputtDist'].flatten()
		edist = edist.reshape(self.data['surface']['nSurfaces'],self.data['surface']['nEnergies'],self.data['surface']['nAngles'])
		eDtotal = np.sum(edist,0).T
		#print(np.nonzero(eDtotal)[0])
                #np.set_printoptions(precision=3)
                #print(eDtotal[eDtotal>0])
                # print(len(edist))
		#eDtotal *= 100.0/eDtotal.max()
                if 'compare-surface' in self.data:
                    copy = eDtotal.copy()
                    copy[copy <= 0] = 0
                    medist = self.data['compare-surface']['surfSputtDist'].flatten()
                    medist = medist.reshape(self.data['compare-surface']['nSurfaces'],self.data['compare-surface']['nEnergies'],self.data['compare-surface']['nAngles'])
                    meDtotal = np.sum(medist,0).T
                    diffdist = meDtotal - eDtotal
                    norm0 = LA.norm(diffdist)
                    norm = LA.norm(eDtotal)
                    print('Sputtering diff. norm',norm0, 'GITR',norm,'percent',norm0/norm*100)
		eDtotal[eDtotal <= 0] = np.nan		# makes areas with zero density white in the plot
		fig = plt.figure(figsize = (10,6))		
		cs = plt.imshow(eDtotal, extent = [0, 1000, 0, 90], cmap='cool', origin = 'lower', aspect = 'auto', interpolation = 'nearest')
		#plt.xscale('log')
		plt.xlabel('Energy [eV]')
		plt.ylabel('Angle [deg]')
		C = plt.colorbar(cs, pad = 0.01, format = '%.3g')
		C.set_label('Particles [a.u.]', rotation = 270, size = FONTSIZE, va = 'bottom')
		plt.title('Sputtering Distribution', fontsize = FONTSIZE)



	def show(self, KEY = None):
		"""
		List the dimensions and variables in the file KEY. Default is: list all available files
		"""
		for file in self.files:
			if KEY is None: print( '\n--------', file,'--------')
			else: file = KEY
			
			arrays,dims,other = [],[],[]
			for key in self.data[file].keys():
				if isinstance(self.data[file][key],np.ndarray): arrays.append(key)
				elif isinstance(self.data[file][key],int): dims.append(key)
				else: other.append(key)
			
			print( 'Dimensions:')
			for key in dims: print( key, '=', self.data[file][key])
		
			print( '\n','Arrays:')
			for key in arrays: print( key, self.data[file][key].shape)

			if len(other) > 0:
				print( '\n','Other Variables:')
				for key in other: print( key, '=', self.data[file][key])
			
			if KEY is not None: break


	def modifyInputTimeSteps(self, nT = 100, path = None):
		"""
		Modify the number of time steps nT in the main input file
		"""
		if path is None: path  = self.path
		self.config['gitrInput']['timeStep']['nT'] = nT
		file = path + 'input/gitrInput.cfg'
		with io.open(file,'w') as f:
			libconf.dump(self.config['gitrInput'],f)


	def plot2dGeom(self, fig = None):
		"""
		Plot the 2D geometry
		"""
		#if not self.config.has_key('gitrGeometry'): 
		if 'gitrGeometry' not in self.config:
			print( 'gitrGeometry.cfg not available')
			return

		x1 = np.array(self.config['gitrGeometry'].geom.x1)	# both notations, with . or with [''] are possible
		x2 = np.array(self.config['gitrGeometry'].geom.x2)
		z1 = np.array(self.config['gitrGeometry'].geom.z1)
		z2 = np.array(self.config['gitrGeometry'].geom.z2)
		Z = np.array(self.config['gitrGeometry'].geom.Z)
	#	length = np.array(self.config['gitrGeometry'].geom.length)
		
		if fig is None: 
			plt.figure(figsize = (9,6))
			plt.title('Geometry', fontsize = FONTSIZE)
			plt.xlabel('x [m]')
			plt.ylabel('z [m]')
		plt.plot(np.append(x1,x1[0]),np.append(z1,z1[0]),'k-',lw = 2)
		plt.xlim(x1.min()-0.05*abs(x1.min()),x1.max()+0.05*abs(x1.max()))
		plt.ylim(z1.min()-0.05*abs(z1.min()),z1.max()+0.05*abs(z1.max()))


	def plot3dGeom(self):
		"""
		Plot the 3D geometry
		"""
		#if not self.config.has_key('gitrGeometry'):
		if 'gitrGeometry' not in self.config:
			print( 'gitrGeometry.cfg not available')
			return

		x1 = np.array(self.config['gitrGeometry'].geom.x1)
		x2 = np.array(self.config['gitrGeometry'].geom.x2)
		x3 = np.array(self.config['gitrGeometry'].geom.x3)
		y1 = np.array(self.config['gitrGeometry'].geom.y1)
		y2 = np.array(self.config['gitrGeometry'].geom.y2)
		y3 = np.array(self.config['gitrGeometry'].geom.y3)
		z1 = np.array(self.config['gitrGeometry'].geom.z1)
		z2 = np.array(self.config['gitrGeometry'].geom.z2)
		z3 = np.array(self.config['gitrGeometry'].geom.z3)
		area = np.array(self.config['gitrGeometry'].geom.area)
		surf = np.array(self.config['gitrGeometry'].geom.surface)
		Z = np.array(self.config['gitrGeometry'].geom.Z)

		xs=[]
		ys=[]
		zs=[]

		ymax = 0.01
		rmax = 0.5
		xmax = 0.06
		xmin = 0.035
		zmax = 0.15
		zmin = 0.0
		xc = 0 #0.0446
		for i in range(0, x1.size - 1):
			#xa = x1[i] - xc
			#xb = x2[i] - xc
			#xc = x3[i] - xc
			#if (math.sqrt(xa*xa + y1[i]*y1[i]) < rmax and 
			#   math.sqrt(xb*xb + y2[i]*y2[i]) < rmax and
			#   math.sqrt(xc*xc + y3[i]*y3[i]) < rmax and
			#if ( z1[i] < zmax and z2[i] < zmax and z3[i] < zmax and
			#   z1[i] > zmin and z2[i] > zmin and z3[i] > zmin and
			#   x1[i] > xmin and x2[i]> xmin and x3[i] > xmin and
			#   y1[i] < ymax and y1[i] > -ymax and 
			#   y2[i] < ymax and y2[i] > -ymax and
			#   y3[i] < ymax and y3[i] > -ymax and
			#   x1[i] < xmax and x2[i] < xmax and x3[i] < xmax ) :

			xs.append(x1[i])
			xs.append(x2[i])
			xs.append(x3[i])
			ys.append(y1[i])
			ys.append(y2[i])
			ys.append(y3[i])
			zs.append(z1[i])
			zs.append(z2[i])
			zs.append(z3[i])
		
		verts = [zip(xs,ys,zs)]
		out = open("gitr-mesh.txt","w")
		for tups in verts:
			for elem in tups:
				out.write(' '.join(str(e) for e in elem) + '\n')
		fig = plt.figure(figsize = (9,6))
		out.close()
		fig = plt.figure(figsize = (9,6))
		ax = fig.add_subplot(111, projection='3d')
		ax.add_collection3d(Poly3DCollection(verts))
		plt.title('3D Geometry', fontsize = FONTSIZE)
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		ax.set_xlim3d(min(xs)-0.01*abs(min(xs)),max(xs)+0.01*abs(max(xs)))
		ax.set_ylim3d(min(ys)-0.01*abs(min(ys)),max(ys)+0.01*abs(max(ys)))
		ax.set_zlim3d(min(zs)-0.01*abs(min(zs)),max(zs)+0.01*abs(max(zs)))
		
		materialSurfaceInidces = np.nonzero(Z)
		surfIndArray = np.asarray(materialSurfaceInidces)
		print( 'Number of W surfaces', surfIndArray.size)
		

	def plotVz(self, N = None):
		"""
		Plot the vertical velocity of N orbits (default is all)
		"""
		#if not self.data.has_key('history'): 
		if 'history' not in self.data:
			print( 'history.nc not available')
			return

		z = self.data['history']['z']
		vz = self.data['history']['vz']

		fig = plt.figure(figsize = (9,6))
		if N is None: N = self.data['history']['nP']
		for i in xrange(N):
			plt.plot(z[i,:],vz[i,:])

		plt.xlabel("z [m]")
		plt.ylabel("v$_z$ [m/s]")
		

	def plotPitch(self):
		"""
		Plot the pitch angle distribution and velocity distributions among the particles
		"""
		#if not self.data.has_key('positions'): 
		if 'positions' not in self.data:
			print( 'positions.nc not available')
			return

		vx = self.data['positions']['vx']
		vy = self.data['positions']['vy']
		vz = self.data['positions']['vz']
		nP = self.data['positions']['nP']
		weights = np.ones(nP)/np.float64(nP)*100
		
		vperp = np.sqrt(vx**2 + vy**2)
		pitchAngle = np.arctan(vperp/vz)
			
		fig = plt.figure(figsize = (9,6))		
		plt.hist(pitchAngle, bins = 30, weights = weights, rwidth = 0.8)
		plt.xlabel("Pitch Angle [rad]")
		plt.ylabel("# Particles [%]")
		plt.xlim(-1.6,1.6)
		
		fig = plt.figure(figsize = (9,12))
		plt.subplot(3,1,1)
		plt.hist(vx, bins = 30, weights = weights, rwidth = 0.8)
		plt.xlabel("v$_x$ [m/s]")
		plt.ylabel("# Particles [%]")
		
		plt.subplot(3,1,2)
		plt.hist(vy, bins = 30, weights = weights, rwidth = 0.8)
		plt.xlabel("v$_y$ [m/s]")
		plt.ylabel("# Particles [%]")
		
		plt.subplot(3,1,3)
		plt.hist(vz, bins = 30, weights = weights, rwidth = 0.8)
		plt.xlabel("v$_z$ [m/s]")
		plt.ylabel("# Particles [%]")
		
		
	def plotErosion(self, logplot = True):
		"""
		Plot the Deposition and Erosion along the surface
		"""
		#if not self.data.has_key('surface'): 
		if 'surface' not in self.data:
			print( 'surface.nc not available')
			return

		length = np.array(self.config['gitrGeometry'].geom.length)
		N = self.data['surface']['nSurfaces']
		surfaces = self.data['surface']['surfaceNumber']*length[0]/np.float64(N)
		grossEro = self.data['surface']['grossErosion']
		grossDep = self.data['surface']['grossDeposition']
		netErosion = grossEro - grossDep
		positiv = netErosion >= 0
		
		top = max([grossEro.max(),grossDep.max(),np.abs(netErosion).max()])
		digit = int(np.ceil(np.log10(top)))
			
		fig = plt.figure(figsize = (9,6))
		if logplot:
			s1 = plt.semilogy(surfaces, grossEro, 'bo', label = 'Gross Erosion')
			s2 = plt.semilogy(surfaces, grossDep, 'ro', label = 'Gross Deposition')
			s3 = plt.semilogy(surfaces[positiv], netErosion[positiv], 'go', label = 'net Erosion')
			s4 = plt.semilogy(surfaces[-positiv], -netErosion[-positiv], 'g^', label = 'net Deposition')
			plt.ylim(1e-2,10**digit)
		else:
			s1 = plt.plot(surfaces, grossEro, 'bo', label = 'Gross Erosion')
			s2 = plt.plot(surfaces, grossDep, 'ro', label = 'Gross Deposition')
			s3 = plt.plot(surfaces, netErosion, 'go', label = 'net Erosion')
			
		plt.legend(prop={'size': 12})
		plt.xlabel("s [m]")
		plt.ylabel('Rate [a.u.]')
		
		
	def plotChargeHistogram(self):
		"""
		Plot the charge distribution among the particles
		"""
		#if not self.data.has_key('positions'): 
		if 'positions' not in self.data:
			print( 'positions.nc not available')
			return
			
		fig = plt.figure(figsize = (9,6))
		bins = np.arange(self.data['positions']['charge'].max()+2)-0.5
		nP = self.data['positions']['nP']
		weights = np.ones(nP)/np.float64(nP)*100
		plt.hist(self.data['positions']['charge'],bins = bins, weights = weights, rwidth = 0.8)
		plt.xlabel('Charge [e]')
		plt.ylabel('# Particles [%]')

		

# ----------------------------------------------------------------------------------------
# --- End of Class -----------------------------------------------------------------------

# --- openNCfile(Filename) ---------------------------------------------------------------

def openNCfile(Filename):
	"""
	Open netcdf file 'Filename' and parse its contents into dictionary
	return dictionary
	"""
	from netCDF4 import Dataset
	from warnings import filterwarnings
	filterwarnings('ignore')
	
	D = {}
	data = Dataset(Filename, 'r')
	
	# Dimensions
	varnames = data.dimensions.keys()
	for n in varnames:
		D[n] = len(data.dimensions[n])
	
	# Variables
	varnames = data.variables.keys()
	for n in varnames:
		if(data.variables[n].size > 1):										# arrays (all types)
			D[n] = np.array(data.variables[n][:])
			if(data.variables[n].dtype == 'S1'):							# character array
				if(data.variables[n].size == data.variables[n].shape[0]):	# 1-D						
					D[n] = ''.join(data.variables[n][:]).strip()
				else:														# 2-D
					D[n] = []
					for i in xrange(data.variables[n].shape[0]):
						D[n].append(''.join(data.variables[n][i,:]).strip())
		else:																# single variable
			if(data.variables[n].dtype in ['float','float32','float64']):	# float
				D[n] = np.float64(data.variables[n][:])
			elif(data.variables[n].dtype in ['int','int32','int64','long']):# int
				D[n] = int(data.variables[n][:])
			elif(data.variables[n].dtype == 'S1'):							# char
				try: D[n] = ''.join(data.variables[n][:])	# at fixed bndy: mgrid_mode exists but is masked -> error
				except: D[n] = data.variables[n][:]
			else:															# unknown
				print( 'unknown datatype in', Filename, ', variable:', n, ' type:',data.variables[n].dtype, 'size:', data.variables[n].size)
				D[n] = data.variables[n][:]
	return D


# -------------------------------------------------------------------------------------------------------------
# --- main ----------------------------------------------------------------------------------------------------
def main():
	g = gitr(show = True)
	plt.show()

if __name__ == '__main__':
	import argparse
	import textwrap
	parser = argparse.ArgumentParser(description = 'Load, Post-process and Plot GITR output', 
				formatter_class = argparse.RawDescriptionHelpFormatter,
				epilog = 'Please report bugs to: wingen@fusion.gat.com')

	parser.add_argument('-v', '--version', help = 'Show current version number and release data', action = 'store_true', default = False)
	args = parser.parse_args()

	if args.version: print( '\ngitr_class, Version: ' + str(_VERSION) + ', Release: ' + _LAST_UPDATE + '\n')
	else: main()



