### Steps for C++ code :

Note: For Mac os install Xquartz first and  for Ubuntu install libx11-dev 

```bash
sudo apt-get install libx11-dev.
```

```bash
git clone https://github.com/davisking/dlib.git

cd dlib 

mkdir  sleep_detection

cd sleep_detection/
```
Paste two files (sleep_detetction.cpp and CMakeLists.txt) here and create build folder 

```bash
mkdir build

cd build 

cmake ..

cmake --build . --config Release
```
Run the app:

```bash
./sleep_detection
```
