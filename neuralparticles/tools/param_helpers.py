#
# Helpers for handling command line parameters and the like
# example: path = getParam("path", "path.uni")
#
import sys
import os
import shutil
import datetime

# global for checking used params
paramUsed = []

# ======================================================================================================================
# read parameters

#! check for a specific parameter, note returns strings, no conversion; not case sensitive! all converted to lower case
def getParam(name, default):
	global paramUsed
	while( len(paramUsed)<len(sys.argv) ):
		paramUsed.append(0);
	for iter in range(1, len(sys.argv)):
		#if(iter <  len(sys.argv)-1): print("Param %s , used %d, val %s" %( sys.argv[iter].lower(), paramUsed[iter] , sys.argv[iter+1]) ); # debug
		if(sys.argv[iter].lower() == name.lower()) and (iter+1<len(paramUsed)):
			paramUsed[iter] = paramUsed[iter+1] = 1;
			return sys.argv[iter+1];
	return default;

def checkUnusedParams():
	global paramUsed
	err = False;
	for iter in range(1, len(sys.argv)):
		if(paramUsed[iter]==0):
			print("Error: param %d '%s' not used!" % (iter,sys.argv[iter]) );
			err = True;
	if err:
		exit(1);

'''def backupSources(name):
	#return; # off
	# save scene file
	#shutil.copyfile( sceneSrcFile, '%s_source.py' % (name) )
	sceneFile = sys.argv[0];
	shutil.copyfile( sceneFile, '%s_scene.py' % (name) )

	# save command line call
	callfile = open( ("%s_call.txt"%name), 'w+')
	callfile.write("\n");
	callfile.write(str(" ".join(sys.argv) ) );
	callfile.write("\n\n");
	callfile.close();'''

# ======================================================================================================================
# others / directory handling

# copies files and config_files into a temporary folder
# returns the path to the temporary folder
def backupSources(data_path):
	tmp_path = data_path + "tmp/"
	if not os.path.exists(tmp_path):
		os.mkdir(tmp_path)
	tmp_path += datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "/"
	os.mkdir(tmp_path)

	print("temporary folder: %s" % tmp_path)

	shutil.copytree("config", tmp_path + "config")
	shutil.copytree(os.path.dirname(sys.argv[0]), tmp_path + "scripts")

	shutil.copy(sys.argv[0], tmp_path)
	
	with open(tmp_path + "call.txt", 'w') as callfile:
		callfile.write(str(" ".join(sys.argv)))

	return tmp_path


# search & create next output dir
def getNextGenericPath(dirPrefix, folder_no = 1, basePath="../data/"):
	test_path_addition = '%s_%04d/' % (dirPrefix, folder_no)
	while os.path.exists(basePath + test_path_addition):
		folder_no += 1
		test_path_addition = '%s_%04d/' % (dirPrefix, folder_no)
		test_folder_no = folder_no
	test_path = basePath + test_path_addition
	print("Using %s dir '%s'" % (dirPrefix, test_path) )
	os.makedirs(test_path)
	return (test_path, folder_no)

def getNextTestPath(folder_no = 1, basePath="../data/"):
	return getNextGenericPath("test", folder_no, basePath)

def getNextSimPath(folder_no = 1, basePath="../data/"):
	return getNextGenericPath("sim", folder_no, basePath)

class Logger(object):
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass    

