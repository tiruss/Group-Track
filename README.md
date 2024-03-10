#  Group Detection and Tracking

## 소개

이 프로젝트는 그룹 클래스 검출에 특화된 YOLOv8 네트워크를 학습하고 이를 이용하여 그룹 검출과 추적을 수행하는 프로젝트입니다.

## 프로젝트 구조


## 설치

```bash
$ git clone https://github.com/tiruss/Group-Track.git
$ cd Group-Track

$ conda create -n group-track python=3.10
$ conda activate group-track

$ pip install boxmot
```

### Download YOLOv8-Group weight

* Download the weight file from
[link](https://drive.google.com/file/d/1vy53-eZITa6hOrp6LQT64J29VSfiGgO-/view?usp=sharing)

* 최상위 디렉토리에 `yolov8-group.pt` 파일을 저장합니다.



## Usage
  
```bash
$ python track.py --source ${source} --yolo-model yolov8-group.pt
```
<details>
<summary>Tracking</summary>

<details>
<summary>Yolo models</summary>

## Detail of models


```bash
$ python tracking/track.py --yolo-model yolov8n       # bboxes only
  python tracking/track.py --yolo-model yolo_nas_s    # bboxes only
  python tracking/track.py --yolo-model yolox_n       # bboxes only
                                        yolov8n-seg   # bboxes + segmentation masks
                                        yolov8n-pose  # bboxes + pose estimation

```

  </details>

<details>
<summary>Tracking methods</summary>

```bash
$ python tracking/track.py --tracking-method deepocsort
                                             strongsort
                                             ocsort
                                             bytetrack
                                             botsort
```

</details>

<details>
<summary>Tracking sources</summary>

Tracking can be run on most video formats

```bash
$ python tracking/track.py --source 0                               # webcam
                                    img.jpg                         # image
                                    vid.mp4                         # video
                                    path/                           # directory
                                    path/*.jpg                      # glob
                                    'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                    'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>Select ReID model</summary>

Some tracking methods combine appearance description and motion in the process of tracking. For those which use appearance, you can choose a ReID model based on your needs from this [ReID model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO). These model can be further optimized for you needs by the [reid_export.py](https://github.com/mikel-brostrom/yolo_tracking/blob/master/boxmot/deep/reid_export.py) script

```bash
$ python tracking/track.py --source 0 --reid-model lmbn_n_cuhk03_d.pt               # lightweight
                                                   osnet_x0_25_market1501.pt
                                                   mobilenetv2_x1_4_msmt17.engine
                                                   resnet50_msmt17.onnx
                                                   osnet_x1_0_msmt17.pt
                                                   clip_market1501.pt               # heavy
                                                   clip_vehicleid.pt
                                                   ...
```

</details>

<details>
<summary>Filter tracked classes</summary>

By default the tracker tracks all MS COCO classes.

If you want to track a subset of the classes that you model predicts, add their corresponding index after the classes flag,

```bash
python tracking/track.py --source 0 --yolo-model yolov8s.pt --classes 16 17  # COCO yolov8 model. Track cats and dogs, only
```

[Here](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/) is a list of all the possible objects that a Yolov8 model trained on MS COCO can detect. Notice that the indexing for the classes in this repo starts at zero

</details>

<details>
<summary>MOT compliant results</summary>

Can be saved to your experiment folder `runs/track/exp*/` by

```bash
python tracking/track.py --source ... --save-mot
```

</details>

</details>

<details>
<summary>Evaluation</summary>

Evaluate a combination of detector, tracking method and ReID model on standard MOT dataset or you custom one by

```bash
# saves dets and embs under ./runs/dets_n_embs separately for each selected yolo and reid model
$ python tracking/generate_dets_n_embs.py --source ./assets/MOT17-mini/train --yolo-model yolov8n.pt yolov8s.pt --reid-model weights/osnet_x0_25_msmt17.pt
# generate MOT challenge format results based on pregenerated detections and embeddings for a specific trackign method
$ python tracking/generate_mot_metrics.py --dets yolov8n --embs osnet_x0_25_msmt17 --tracking-method botsort
# uses TrackEval to generate MOT metrics for the tracking results under ./runs/mot/<dets+embs+tracking-method>
$ python tracking/val.py --benchmark MOT17-mini --dets yolov8n --embs osnet_x0_25_msmt17 --tracking-method botsort
```

</details>


<details>
<summary>Evolution</summary>

We use a fast and elitist multiobjective genetic algorithm for tracker hyperparameter tuning. By default the objectives are: HOTA, MOTA, IDF1. Run it by

```bash
# saves dets and embs under ./runs/dets_n_embs separately for each selected yolo and reid model
$ python tracking/generate_dets_n_embs.py --source ./assets/MOT17-mini/train --yolo-model yolov8n.pt yolov8s.pt --reid-model weights/osnet_x0_25_msmt17.pt
# evolve parameters for specified tracking method using the selected detections and embeddings generated in the previous step
$ python tracking/evolve.py --benchmark MOT17-mini --dets yolov8n --embs osnet_x0_25_msmt17 --n-trials 9 --tracking-method botsort
```

The set of hyperparameters leading to the best HOTA result are written to the tracker's config file.

</details>

