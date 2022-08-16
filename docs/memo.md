## cmd 
    ps aux|grep train|awk '{print $2}'|xargs kill -9
    fuser -v /dev/nvidia0 | awk '{print}' | xargs kill -9
    grep -rl "string" /path
    du -h --max-depth=1 |grep [M]
    fuser -v /dev/
    sof /dev/nvidia*.
    cat ~/include/cudnn.h | grep CUDNN_MAJOR -A
    ffmpeg -pix_fmt rgb24 -framerate 1 -f image2 -i %02d.png name.gif
    ffmpeg -pix_fmt rgb24 -framerate 1 -pattern_type glob -i '*.png' name.gif
    find src_dir/ -name "*.png" -print0 | xargs -r0 mv -t dst_dir
    tmux attach -d -t cl 

## mount 
    sudo mount -t nfs -o vers=3,nolock,proto=tcp,noresvport src_path dst_path
    sudo apt install nfs-common

## bash
    HOME=/mnt/workspace
    CUDA=cuda-10.2
    export CUDA_HOME=$Home$/$CUDA$
    export LD_LIBRARY_PATH=$Home$/lib/:$Home$/lib64/:$Home$/$CUDA$/lib64/:$LD_LIBRARY_PATH
    export PATH=$Home$/bin:$Home$/anaconda2/bin:$Home$/$CUDA$/bin:$PATH
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

## tmux
    git clone https://github.com/gpakosz/.tmux
    ln -s ~/.tmux/.tmux.conf .tmux.conf 
    
## nas
    apt-get install nfs-common
    sudo yum install nfs-utils

## vim    
    git clone --depth=1 https://github.com/amix/vimrc.git ~/.vim_runtime
    sh ~/.vim_runtime/install_awesome_vimrc.sh

## git 
    git config --global user.name "name"
    git config --global user.email "email"

## Install Docker
    sudo yum list installed | grep "docker"
    sudo curl -s -L https://download.docker.com/linux/centos/docker-ce.repo | sudo tee /etc/yum.repos.d/docker-ce.repo
    sudo yum list docker-ce --showduplicates | sort -r  
    sudo yum install docker-ce-3:19.03.8-3.el7    
    sudo service docker restart

## Install nvidia Docker
    sudo curl -s -L https://nvidia.github.io/nvidia-docker/centos7/nvidia-docker.repo   |  sudo tee /etc/yum.repos.d/nvidia-docker.repo
    sudo yum install -y nvidia-docker2-2.2.0-1

## Run Docker
    sudo systemctl --now enable docker 
    systemctl stop docker
    sudo docker images 
    sudo docker rmi 
    sudo docker build -t name .
    sudo docker rm $(sudo docker ps -a -q)
    sudo docker tag id url:tag
    sudo docker push url:tag    
    sudo docker run -ti -v src_dir:dst_dir id /bin/bash
