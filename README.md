# NS-3 Installation Guide (Mac, Windows, Linux)

## **1. Prerequisites**

### **Mac (Homebrew)**

Install Homebrew:

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Install required packages:

```sh
brew install cmake pkg-config python@3.11 boost gcc
```

### **Linux (Ubuntu/Debian)**

Install required packages:

```sh
sudo apt update && sudo apt upgrade -y
sudo apt install -y git cmake pkg-config python3 python3-dev python3-pip python3-setuptools python3-wheel \
    g++ gcc gdb valgrind doxygen graphviz imagemagick texlive texlive-latex-extra texlive-fonts-recommended \
    python3-sphinx python3-matplotlib python3-numpy libboost-all-dev
```

### **Windows (WSL - Ubuntu)**

If you are using Windows, install **WSL (Ubuntu)** and follow the **Linux** instructions above.

## **2. NS-3 Installation**

### **Clone NS-3 Repository**

```sh
git clone https://gitlab.com/nsnam/ns-3-dev.git
cd ns-3-dev
```

### **Configure NS-3 with Python Bindings**

```sh
./ns3 configure --enable-examples --enable-tests --enable-python-bindings
```

### **Build NS-3**

```sh
./ns3 build
```

## **3. Environment Setup**

### **Set Up Environment Variables**

Add the following to your shell configuration file:

#### **Mac & Linux** (`~/.zshrc` or `~/.bashrc`)

```sh
# NS-3 Environment Variables
export NS3_HOME=~/ns-3-dev
export NS3_BUILD=$NS3_HOME/build
export PYTHONPATH=$PYTHONPATH:$NS3_BUILD/bindings/python
```

Apply changes:

```sh
source ~/.zshrc  # or source ~/.bashrc
```

#### **Windows (WSL - Ubuntu)**

Same as Linux, add the above lines to `~/.bashrc` and apply with:

```sh
source ~/.bashrc
```

## **4. Verification**

### **Test NS-3 Installation**

```sh
./test.py
```

### **Test Python Bindings**

```sh
python3
>>> from ns import ns
```

If no errors appear, the setup is successful! 🚀
