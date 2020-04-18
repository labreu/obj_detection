## Fontes de dados
### Robos
- https://imagetagger.bit-bots.de/images/imageset/610/

### Conectar com Remote Development do VS Code na maquina

```shell

git clone https://github.com/tensorflow/models.git
pip3 install --ignore-installed --upgrade tensorflow-gpu==1.14

sudo apt-get install protobuf-compiler python-pil python-lxml python-tk -y
pip3 install --user Cython
pip3 install --user contextlib2
pip3 install --user jupyter
pip3 install --user matplotlib

```




```shell
cd models/research/
source ~/.bashrc
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip

./bin/protoc object_detection/protos/*.proto --python_out=.

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python3 object_detection/builders/model_builder_test.py
```

## Treinando custom classifier
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html


#### Baixar partition_dataser.py do site https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html

```shell
wget https://objdetectionfei.s3.amazonaws.com/imageset_610.zip

python3 clean_unused_imgs.py 
python3 partition_dataser.py -x -i images/ -r 0.1

mkdir annotations
vim annotations/label_map.pbtxt
```

item {
    id: 1
    name: ‘ball'
}

```shell
python3 xml_to_csv.py -i images/train -o annotations/train_labels.csv
python3 xml_to_csv.py -i images/test -o annotations/test_labels


python3 generate_tfrecord.py --label=ball --csv_input=annotations/train_labels.csv --output_path=annotations/train.record --img_path=images/train

python3 generate_tfrecord.py --label=ball --csv_input=annotations/test_labels.csv --output_path=annotations/test.record --img_path=images/test
```


#### Ver modelos em: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models-coco-models

```shell
wget link_modelo
tar -xzvf ssd_mobilenet_v1_coco.tar.gz
```


Rename do modelo:
```shell
mv ssd_inception_v2_coco_2018_01_28/ pre-trained-model
```

E tambem o arquivo de config: https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs
```shell
wget link_conf
```

#### No arquivo de config mudar
- num_classes: 1 na linha 9 na model/ssd
- type: ‘ssd_inception_v2’ na linha 77 nas feature_extractor
- batch_size: 12 na linha 136 nas train configs se houver mais memória
- fine_tune_checkpoint: "pre-trained-model/model.ckpt” nas train linha 151
- num_steps: 200000 na linha 157 nas train pra ver quantidade de steps

train_input_reader: {
  tf_record_input_reader {
    input_path: "annotations/train.record"
  }
  label_map_path: "annotations/label_map.pbtxt"
}

eval_config: {
  num_examples: 8000
  #Note: The below line limits the evaluation process to 10 evaluations.
  #Remove the below line to evaluate indefinitely.
  max_evals: 10
  metrics_set: "coco_detection_metrics"
}

eval_input_reader: {
  tf_record_input_reader {
    input_path: "annotations/test.record"
  }
  label_map_path: "annotations/label_map.pbtxt"
  shuffle: false
  num_readers: 1
}


## Configurar COCO
```shell
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools/ ~/models/research 
```
#original era: cp -r pycocotools <PATH_TO_TF>/TensorFlow/models/research/


## Treinando
```shell
cd ~/models/research
cp object_detection/model_main.py .

python3 model_main.py --alsologtostderr --model_dir=. --pipeline_config_path=ssd_inception_v2_coco.config
```

## Obs

Model to reach a TotalLoss of at least 2 (ideally 1 and lower) if you want to achieve “fair” detection results.

While the evaluation process is running, it will periodically (every 300 sec by default) check and use the latest training/model.ckpt-* checkpoint files to evaluate the performance of the model. The results are stored in the form of tf event files (events.out.tfevents.*) inside training/eval_0. These files can then be used to monitor the computed metrics, using the process described by the next section.

