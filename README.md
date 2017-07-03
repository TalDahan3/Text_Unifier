# torch-rnn
torch-rnn provides high-performance, reusable RNN and LSTM modules for torch7, and uses these modules for character-level
language modeling similar to [char-rnn](https://github.com/karpathy/char-rnn).

You can find documentation for the RNN and LSTM modules [here](doc/modules.md); they have no dependencies other than `torch`
and `nn`, so they should be easy to integrate into existing projects.

Compared to char-rnn, torch-rnn is up to **1.9x faster** and uses up to **7x less memory**. For more details see 
the [Benchmark](#benchmarks) section below.


# Installation

## Docker Images
Cristian Baldi has prepared Docker images for both CPU-only mode and GPU mode;
you can [find them here](https://github.com/crisbal/docker-torch-rnn).

## System setup
You'll need to install the header files for Python 2.7 and the HDF5 library. On Ubuntu you should be able to install
like this:

```bash
sudo apt-get -y install python2.7-dev
sudo apt-get install libhdf5-dev
```

## Python setup
The preprocessing script is written in Python 2.7; its dependencies are in the file `requirements.txt`.
You can install these dependencies in a virtual environment like this:

```bash
virtualenv .env                  # Create the virtual environment
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install Python dependencies
# Work for a while ...
deactivate                       # Exit the virtual environment
```

## Lua setup
The main modeling code is written in Lua using [torch](http://torch.ch); you can find installation instructions
[here](http://torch.ch/docs/getting-started.html#_). You'll need the following Lua packages:

- [torch/torch7](https://github.com/torch/torch7)
- [torch/nn](https://github.com/torch/nn)
- [torch/optim](https://github.com/torch/optim)
- [lua-cjson](https://luarocks.org/modules/luarocks/lua-cjson)
- [torch-hdf5](https://github.com/deepmind/torch-hdf5)

After installing torch, you can install / update these packages by running the following:

```bash
# Install most things using luarocks
luarocks install torch
luarocks install nn
luarocks install optim
luarocks install lua-cjson

# We need to install torch-hdf5 from GitHub
git clone https://github.com/deepmind/torch-hdf5
cd torch-hdf5
luarocks make hdf5-0-0.rockspec
```

### CUDA support (Optional)
To enable GPU acceleration with CUDA, you'll need to install CUDA 6.5 or higher and the following Lua packages:
- [torch/cutorch](https://github.com/torch/cutorch)
- [torch/cunn](https://github.com/torch/cunn)

You can install / update them by running:

```bash
luarocks install cutorch
luarocks install cunn
```

## OpenCL support (Optional)
To enable GPU acceleration with OpenCL, you'll need to install the following Lua packages:
- [cltorch](https://github.com/hughperkins/cltorch)
- [clnn](https://github.com/hughperkins/clnn)

You can install / update them by running:

```bash
luarocks install cltorch
luarocks install clnn
```

## OSX Installation
Jeff Thompson has written a very detailed installation guide for OSX that you [can find here](http://www.jeffreythompson.org/blog/2016/03/25/torch-rnn-mac-install/).


