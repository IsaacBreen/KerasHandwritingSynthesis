import os

if __name__=="__main__":
	# Download data
	os.system(" wget https://storage.googleapis.com/isaacs_space/original_part.zip original_part.zip")
	os.system("unzip data/original_part.zip > /dev/null")
	os.system("rm data/original_part.zip")
