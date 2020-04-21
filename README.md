## Fontes de dados
### Robos
- https://imagetagger.bit-bots.de/images/imageset/610/


## Para AWS
Deep Learning AMI (Ubuntu 16.04) Version 27.0 - ami-0a79b70001264b442

### Conectar com Remote Development do VS Code na maquina

```shell
source activate tensorflow_p36

git clone https://github.com/tensorflow/models.git
cd models/research/
source ~/.bashrc
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip

./bin/protoc object_detection/protos/*.proto --python_out=.

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python3 object_detection/builders/model_builder_test.py
```

ou


```shell
pip3 install --ignore-installed --upgrade tensorflow-gpu==1.14
pip3 install --user Cython
pip3 install --user contextlib2
pip3 install --user jupyter
pip3 install --user matplotlib

```


```shell
cd
git clone https://github.com/lucasrabreu/obj_detection
cd obj_detection
mkdir images
cd images
wget https://objdetectionfei.s3.amazonaws.com/imageset_610.zip
unzip imageset_610.zip
wget https://objdetectionfei.s3.amazonaws.com/labels.zip
unzip labels.zip
```

## Treinando custom classifier
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html


#### Baixar partition_dataser.py do site https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html

```shell
cd ..

python3 clean_unused_imgs.py --base images
python3 partition_dataser.py -x -i images/ -r 0.1
mkdir annotations
vim annotations/label_map.pbtxt

```

item { 
    id: 1 
    name: ‘ball'
} 

item {
    id: 2
    name: 'dog'
}

```shell
python3 xml_to_csv.py -i images/train -o annotations/train_labels.csv
python3 xml_to_csv.py -i images/test -o annotations/test_labels.csv

python3 generate_tfrecord.py --label=ball --csv_input=annotations/train_labels.csv --output_path=annotations/train.record --img_path=images/train

python3 generate_tfrecord.py --label=ball --csv_input=annotations/test_labels.csv --output_path=annotations/test.record --img_path=images/test

###########


python3 generate_tfrecord.py --label0=ball --label1=robot --csv_input=annotations/train_labels.csv --output_path=annotations/train.record --img_path=images/train

python3 generate_tfrecord.py  --label0=ball --label1=robot --csv_input=annotations/test_labels.csv --output_path=annotations/test.record --img_path=images/test
```


#### Ver modelos em: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models-coco-models

```shell
wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz

tar -xzvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz

mv ssdlite_mobilenet_v2_coco_2018_05_09 pre-trained-model
```


E tambem o arquivo de config: https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs
```shell
#ja tem no repo wget https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssdlite_mobilenet_v2_coco.config

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

### Opcoes de data augmentation
https://github.com/tensorflow/models/blob/master/research/object_detection/builders/preprocessor_builder_test.py


## Configurar COCO
```shell
cd
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cd
cp -r cocoapi/PythonAPI/pycocotools/ models/research/
```

## Treinando
```shell
cd obj_detection
cp models/research/object_detection/model_main.py obj_detection/

python3 model_main.py --alsologtostderr --model_dir=newmodel --pipeline_config_path=ssdlite_mobilenet_v2_coco.config
```

## Obs

Model to reach a TotalLoss of at least 2 (ideally 1 and lower) if you want to achieve “fair” detection results.

While the evaluation process is running, it will periodically (every 300 sec by default) check and use the latest training/model.ckpt-* checkpoint files to evaluate the performance of the model. The results are stored in the form of tf event files (events.out.tfevents.*) inside training/eval_0. These files can then be used to monitor the computed metrics, using the process described by the next section.

### Export Inference graph
```shell
cd
cp models/research/object_detection/export_inference_graph.py obj_detection/
```

Ver maior model.ckpt-XXXX.meta

```shell
python3 export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=ssdlite_mobilenet_v2_coco.config \
    --output_directory=. \
    --trained_checkpoint_prefix=model.ckpt-32376

ls -laht

python3 export_tflite_ssd_graph.py \
    --pipeline_config_path ssdlite_mobilenet_v2_coco.config \
    --trained_checkpoint_prefix model.ckpt-32376 \
    --output_directory tflite

```

# Predict
```shell
python3 predict_boxes.py
```
