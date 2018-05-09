MLSGPU is software that constructs a triangulated mesh from a point cloud. It is designed to handle massive point clouds (billions of points), and uses OpenCL for GPU acceleration. It implements the method described in the paper <cite>[Moving Least-Squares Reconstruction of Large Models with GPUs](https://ieeexplore.ieee.org/document/6589586/)</cite>.

The documentation can be found at http://bmerry.github.io/mlsgpu. For the very impatient (who have all the dependencies installed), the installation process is

```sh
python waf configure --variant=release
python waf
sudo python waf install
```
