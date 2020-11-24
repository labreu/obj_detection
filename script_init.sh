nohup python3 model_main.py --alsologtostderr --model_dir=teste3novembromobnetv2metric --pipeline_config_path=newconfigmobnetv2.config &
nohup tensorboard --logdir=./teste3novembromobnetv2metric &

source activate tensorflow_p36
cd
cd models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd
cd obj_detection/


