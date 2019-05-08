import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import sys

def read_data(target_folder,cur_t):
	return np.load("{}/output{}.npy".format(target_folder,cur_t)).T
	
def get_projection(data3d,camera,angle,plane):
	"""Projects the 3d points onto a 2d canvas, defined by the location and angle of a camera and its film"""
	##Arguments:
	#  data3d - a 3xn_points numpy array, giving the x,y,z coordinates of each point in the dataset
	#  camera - a 3x1 numpy column vector, giving the location of the camera in the same coordinates
	#  angle  - a 3x1 numpy column vector, giving the rotations applied to the camera.
	#			This is easiest to think about like so:
	#			The camera begins pointed directly up (i.e. a unit vector at whatever x,y,z specified, but pointed parallel to the z axis)
	#			The film/canvas is at the tip of the unit vector, and its internal x,y coordinate system begins aligned with
	#			the data's coordinate system
	#			The three entries of angle, call them alpha, beta, and gamma, define how the camera is rotated in space
	#			  alpha is applied first, and rotates the camera around (a line parallel to) the x-axis (anti-clockwise if you're standing at x= positive infinity)
	#			  beta is applied second, and rotates the resulting vector around (a line parallel to) the y-axis, again anti-clockwise
	#			  gamma is applied last, and rotates the result of both moves above around the z axis
	#			For example a rotation of [-pi/2,0,0] leaves the camera pointed along the positive y axis, but "upside down",
	#			in that if the data all move 1 unit up, the image on the canvas moves down, (i.e. the y coordinates in-image decrease)
	#			None of the entries rotate the camera "left" or "right" or "up" or "down", so it can be difficult to
	#			adjust an existing shot. I literally suggest taping some cardboard to a dry erase marker and marking out
	#			image-internal x,y coordinates as well as absolute x,y,z coordinates in space.
	#  plane - Defines the offset of the image plane from the camera. The first two entries should be left as 0, and the
	#		   final entry defines the level of zoom. 1 is a fine default and 10 will zoom in on the image by a factor of 10.
	#
	##Notes:
	#  This implements a perspective projection, as if a real movie camera were present. Objects near the camera 
	#  will appear forshortened, etc. For an orthographic projection, multiply both the distance from the subject
	#  and the zoom (final entry of 'plane') by e.g. 100

	
	##
	# Define the 3x3 coordinate rotation matrices
	##
	theta = angle[2]
	rz =np.array([
		[np.cos(theta),	 np.sin(theta), 0],
		[-np.sin(theta), np.cos(theta), 0],
		[0,				 0,				1]
	]) 

	theta = angle[1]
	ry =np.array([
		[np.cos(theta), 0, -np.sin(theta)],
		[0,				1, 0			 ],
		[np.sin(theta), 0, np.cos(theta) ]
	])
	theta = angle[0]
	rx =np.array([
		[1, 0,			   0			],
		[0, np.cos(theta), np.sin(theta)],
		[0,-np.sin(theta), np.cos(theta)]
	])
	rmat = np.dot(rx,np.dot(ry,rz))
	
	def to_camera_coords():
		"""Switch the data into the camera's coordinate system"""
		# in-system, the camera is at 0,0,0, it's aimed along the positive z axis, and
		# the x,y axes are the horizontal and verticals of the image the camera is seeing 
		disp = data3d - camera
		d = np.dot(rmat,disp)
		return d
	
	def drop_points(d):
		"""Drop all points that are behind the camera"""
		d = d[:,d[z,:]>0] # drop points on wrong side of screen
		return d
	
	# project to view plane
	def project():
		"""Project out the points' z values"""
		# note that we're not just dropping the z values, we're getting true perspective
		bx = plane[z]/d[z,:]*d[x,:]+plane[x]
		by = plane[z]/d[z,:]*d[y,:]+plane[y]
	
		return bx,by
	
	x,y,z = (0,1,2) #aliases so we can write slice along x,y,z instead of numbers
	d=to_camera_coords()
	d = drop_points(d)
	
	bx,by = project()
	
	return bx,by

def pretty_plot(proj_x,proj_y,ax):
	nBins = 500
	xLow=-1; xHigh=1
	yLow=-1; yHigh=1
	
 
	# get a raster grid counting the number of particles at each pixel
	obj_count,_,_ = np.histogram2d(proj_x,proj_y, bins=nBins, range=[[xLow,xHigh],[yLow,yHigh]])
	im=ax.imshow(np.log(obj_count+1).T,origin='lower')
	
	# equivalent, direct code (but harder to move to GPU, and hist2d is one of
	# the expensive operations)
	#im = ax.hist2d(proj_x,proj_y, bins=nBins, range=[[xLow,yHigh],[yLow,yHigh]], norm=LogNorm());
	
	ax.set_facecolor(plt.cm.viridis(0))
	ax.set_aspect('equal')
	return im
	
	

def main(target_folder, max_timestep):
	
	fig,ax = plt.subplots(1,1,figsize=(12,12))
	for cur_t in range(max_timestep):
		print("Processing {} of {}".format(cur_t+1,max_timestep))
		
		directory_name = "output_images{}".format(cur_t)
		if not os.path.exists(directory_name):
			os.mkdir(directory_name)
	
		data = read_data(target_folder,cur_t)
		
		data_center = np.mean(data,axis=1).reshape(-1,1)

		#rotating around the z axis
		count=0
		for rad in np.linspace(0,2*np.pi,40):
			distance = 6
			offset = np.array([[np.sin(rad)*distance],[-np.cos(rad)*distance],[0]])
			camera = data_center + offset
			gamma = rad

			angle = np.array([-np.pi/2,np.pi,gamma])
			plane = np.array([0,0,1])

			proj_x,proj_y = get_projection(data,camera,angle,plane)
			im = pretty_plot(proj_x,proj_y,ax)
			def save_it():
				plt.savefig("{}/image{}.png".format(directory_name,count))
			save_it()
			count+=1
			ax.clear()

if __name__ == "__main__":
	argv = sys.argv
	added_args = len(argv)-1
	if added_args<1 or added_args>3:
		print("Usage: extract_flybys_cpu [target_folder] [max_timestep] [profile?] ")
		exit()
		
	# target folder
	try:
		folder_name = argv[1]
	except:
		folder_name = os.getcwd()
	
	# max timestep
	if added_args >= 2:
		max_time_string = argv[2]
	else:
		max_time_string = "256"
	
	try:
		max_timestep = int(max_time_string)
	except:
		print("Error: couldn't parse first input [max_timestep]")
		exit()
		
	
	# profiling flag
	if added_args >= 3:
		profile_string = argv[3]
	else:
		profile_string = "0"
	
	try:
		profile = int(profile_string)
	except:
		print("Error: couldn't parse third input [profile?]")
		exit()
	
	# call the code
	if profile:
		print("Profiling")
		import cProfile
		print(cProfile.run("main(folder_name, max_timestep)"))
	else:
		main(folder_name, max_timestep)
	
	