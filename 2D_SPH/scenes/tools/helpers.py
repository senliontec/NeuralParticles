# 
# tool runs helpers
#
import sys

# ======================================================================================================================
# duplicated from main "helpers", not to require manta

def getParam(name, default, paramUsed):
	while( len(paramUsed)<len(sys.argv) ):
		paramUsed.append(0);
	for iter in range(1, len(sys.argv)):
		#if(iter <  len(sys.argv)-1): print("Param %s , used %d, val %s, name %s " %( sys.argv[iter].lower(), paramUsed[iter] , sys.argv[iter+1], name) ); # debug
		if(sys.argv[iter].lower() == name) and (iter+1<len(paramUsed)):
			paramUsed[iter] = paramUsed[iter+1] = 1;
			return sys.argv[iter+1];
	return default;

def getColParam(name, default, paramUsed):
	while( len(paramUsed)<len(sys.argv) ):
		paramUsed.append(0);
	for iter in range(1, len(sys.argv)):
		#if(iter <  len(sys.argv)-1): print("Param %s , used %d, val %s, name %s " %( sys.argv[iter].lower(), paramUsed[iter] , sys.argv[iter+1], name) ); # debug
		if(sys.argv[iter].lower() == name) and (iter+1<len(paramUsed)):
			paramUsed[iter] = paramUsed[iter+1] = 1;
			paramUsed[iter+2] = paramUsed[iter+3] = 1;
			return [ float(sys.argv[iter+1]), float(sys.argv[iter+2]), float(sys.argv[iter+3]) ]
	return default;

def checkUnusedParam(paramUsed, off=0):
	err = False;
	for iter in range(1+off, len(sys.argv)):
		if(paramUsed[iter]==0):
			print("Error: param %d '%s' not used!" % (iter,sys.argv[iter]) );
			err = True;
	if err:
		exit(1);



