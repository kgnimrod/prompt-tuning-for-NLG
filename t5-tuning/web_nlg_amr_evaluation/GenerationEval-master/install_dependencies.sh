# INSTALL PYTHON DEPENDENCIES
pip3 install -r requirements.txt

# INSTALL BLEURT
pip3 install --upgrade pip
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip3 install . --user
wget https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip
unzip bleurt-base-128.zip
rm bleurt-base-128.zip 
cd ../
mv bleurt metrics

# INSTALL METEOR
wget https://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz
tar -xvf meteor-1.5.tar.gz
mv meteor-1.5 metrics
rm meteor-1.5.tar.gz

#CHECKPOINT_URLS = {
#    "bleurt-tiny-128": "https://storage.googleapis.com/bleurt-oss/bleurt-tiny-128.zip",
#    "bleurt-tiny-512": "https://storage.googleapis.com/bleurt-oss/bleurt-tiny-512.zip",
#    "bleurt-base-128": "https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip",
#    "bleurt-base-512": "https://storage.googleapis.com/bleurt-oss/bleurt-base-512.zip",
##    "bleurt-large-128": "https://storage.googleapis.com/bleurt-oss/bleurt-large-128.zip",
#    "bleurt-large-512": "https://storage.googleapis.com/bleurt-oss/bleurt-large-512.zip",
#    "BLEURT-20-D3": "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D3.zip",
#    "BLEURT-20-D6": "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D6.zip",
#    "BLEURT-20-D12": "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip",
#    "BLEURT-20": "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip",
#}