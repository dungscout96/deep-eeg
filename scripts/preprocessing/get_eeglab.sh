#! /bin/bash

#
# Clone eeglab repo and download Biosig plugin
# for ASR preprocessing of Mahnob
#

git clone --recurse-submodules -j8 https://github.com/sccn/eeglab.git
cd eeglab/plugins
curl https://sccn.ucsd.edu/eeglab/plugins/BIOSIG3.8.1.zip -o BIOSIG3.8.1.zip
unzip BIOSIG3.8.1.zip -d BIOSIG3.8.1
rm BIOSIG3.8.1.zip
