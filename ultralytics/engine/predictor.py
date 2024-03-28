# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ yolo mode=predict model=yolov8n.pt source=0                               # webcam
                                                img.jpg                         # image
                                                vid.mp4                         # video
                                                screen                          # screenshot
                                                path/                           # directory
                                                list.txt                        # list of images
                                                list.streams                    # list of streams
                                                'path/*.jpg'                    # glob
                                                'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP, TCP stream

Usage - formats:
    $ yolo mode=predict model=yolov8n.pt                 # PyTorch
                              yolov8n.torchscript        # TorchScript
                              yolov8n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                              yolov8n_openvino_model     # OpenVINO
                              yolov8n.engine             # TensorRT
                              yolov8n.mlpackage          # CoreML (macOS-only)
                              yolov8n_saved_model        # TensorFlow SavedModel
                              yolov8n.pb                 # TensorFlow GraphDef
                              yolov8n.tflite             # TensorFlow Lite
                              yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov8n_paddle_model       # PaddlePaddle
"""
import json
import platform
import threading
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox, classify_transforms
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
from ultralytics.utils.checks import check_imgsz, check_imshow
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import select_device, smart_inference_mode

STREAM_WARNING = """
WARNING ⚠️ inference results will accumulate in RAM unless `stream=True` is passed, causing potential out-of-memory
errors for large sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict/ for help.

Example:
    results = model(source=..., stream=True)  # generator of Results objects
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs
"""


#그룹 pair
def group_pairs(bbox_cls_0, bbox_cls_1, bbox_id_0, bbox_id_1):
    matches = []
    
    for person_idx, person_box in enumerate(bbox_cls_0):
        best_iou = -1  # 가장 좋은 IoU 값을 추적
        best_group_id = -1  # 매칭되지 않는 경우를 처리하기 위해 초기값 -1 설정

        for group_idx, group_box in enumerate(bbox_cls_1):
            iou = calculate_iou(person_box, group_box)  # IoU 계산

            # IoU가 일정 기준치 이상인 경우에만 매칭을 고려
            if iou > best_iou and iou > 0.2:
                best_iou = iou
                best_group_id = bbox_id_1[group_idx]

        # 최종 매칭 리스트에 추가
        matches.append((int(bbox_id_0[person_idx]), int(best_group_id)))
    
    return matches


#iou계산
def calculate_iou(box1, box2):
    # Unpack the positions
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    inter_width = min(x1_max, x2_max) - max(x1_min, x2_min)
    inter_height = min(y1_max, y2_max) - max(y1_min, y2_min)
    
    if inter_width <= 0 or inter_height <= 0:
        # No overlap
        return 0.0
    
    intersection_area = inter_width * inter_height
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou


class BasePredictor:
    """
    BasePredictor.

    A base class for creating predictors.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_warmup (bool): Whether the predictor has finished setup.
        model (nn.Module): Model used for prediction.
        data (dict): Data configuration.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_path (str): Path to video file.
        vid_writer (cv2.VideoWriter): Video writer for saving video output.
        data_path (str): Path to data.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)
        
     
        self.previous_frame_data = {}
        self.current_frame=0
        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer, self.vid_frame = None, None, None
        self.plotted_img = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        self._lock = threading.Lock()  # for automatic thread-safe inference
        self.ori_group_id=[]
        self.new_ori_group_id=[]
        self.new_gid=1
        self.missing_frame_counts={}
        self.prev_matches = None
        self.thresold=30

        callbacks.add_integration_callbacks(self)

    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def inference(self, im, *args, **kwargs):
        """Runs inference on a given image using the specified model and arguments."""
        visualize = increment_path(self.save_dir / Path(self.batch[0][0]).stem,
                                   mkdir=True) if self.args.visualize and (not self.source_type.tensor) else False
        return self.model(im, augment=self.args.augment, visualize=visualize)

    def pre_transform(self, im):
        """
        Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
            (list): A list of transformed images.
        """
        same_shapes = all(x.shape == im[0].shape for x in im)
        letterbox = LetterBox(self.imgsz, auto=same_shapes and self.model.pt, stride=self.model.stride)
        return [letterbox(image=x) for x in im]
    
    def write_results(self, idx, results, batch):
        """Write inference results to a file or directory."""
     
        p, im, _ = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.source_type.webcam or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        result = results[idx]
        log_string += result.verbose()

        if self.args.save or self.args.show:  # Add bbox to image
            plot_args = {
                'line_width': self.args.line_width,
                'boxes': self.args.show_boxes,
                'conf': self.args.show_conf,
                'labels': self.args.show_labels,
                'group_count': self.previous_frame_data,
                'group_list': self.all_group,
                'img_count':self.dataset.count
                }
            if not self.args.retina_masks:
                plot_args['im_gpu'] = im[idx]
            self.plotted_img = result.plot(**plot_args)
          


        return log_string

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions for an image and returns them."""
        return preds


    
    
    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        """Performs inference on an image or stream."""
        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Result into one

    def predict_cli(self, source=None, model=None):
        """
        Method used for CLI prediction.

        It uses always generator as outputs as not required by CLI mode.
        """
        gen = self.stream_inference(source, model)
        for _ in gen:  # running CLI inference without accumulating any outputs (do not modify)
            pass

    def setup_source(self, source):
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = getattr(self.model.model, 'transforms', classify_transforms(
            self.imgsz[0])) if self.args.task == 'classify' else None
        self.dataset = load_inference_source(source=source,
                                             imgsz=self.imgsz,
                                             vid_stride=self.args.vid_stride,
                                             buffer=self.args.stream_buffer)
        self.source_type = self.dataset.source_type
        if not getattr(self, 'stream', True) and (self.dataset.mode == 'stream' or  # streams
                                                  len(self.dataset) > 1000 or  # images
                                                  any(getattr(self.dataset, 'video_flag', [False]))):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_path = [None] * self.dataset.bs
        self.vid_writer = [None] * self.dataset.bs
        self.vid_frame = [None] * self.dataset.bs
        
    
    def update_group_pair_counts(self, matches):
        self.current_frame += 1
        prev_frame = self.previous_frame_data.copy()
    
        # 현재 프레임에서 새롭게 발견된 key를 추가하고, 존재하는 key의 카운트를 업데이트합니다.
        for key in matches:
            if key in prev_frame:
                if key[1] > 0:
                # 이미 존재하는 key의 카운트를 증가시킵니다. 최대값은 60입니다.
                    self.previous_frame_data[key] = min(prev_frame[key] + 1, 60)
            else:
                # 새로운 key를 추가합니다.
                self.previous_frame_data[key] = 1
                
        keys_to_delete = [] 
        # 이전 프레임에 있었지만 현재 프레임에서 사라진 key에 대한 처리
        if self.current_frame >=60:
            
            for key in prev_frame:
                if key not in matches:
                    self.previous_frame_data[key] = max(prev_frame[key] - 1, 0)
                    if self.previous_frame_data[key] == 0:
                        keys_to_delete.append(key)
                else:
                    if key[1] not in self.ori_group_id and self.previous_frame_data[key]>=self.thresold:
                        self.ori_group_id.append(key[1])
                        self.new_ori_group_id.append(self.new_gid)
                        self.new_gid +=1  
            for key in keys_to_delete:
                del self.previous_frame_data[key]

        # 데이터 정렬
        sorted_keys = sorted(self.previous_frame_data.keys(), reverse=True)
        self.previous_frame_data = {key: self.previous_frame_data[key] for key in sorted_keys}

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
      
        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            LOGGER.info('')
        
        # Setup model
        if not self.model:
            self.setup_model(model)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # Warmup model
            if not self.done_warmup:
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True

            self.seen, self.windows, self.batch, profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile())
            self.run_callbacks('on_predict_start')
            
            for batch in self.dataset:
                self.run_callbacks('on_predict_batch_start')
                self.batch = batch
                path, im0s, vid_cap, s = batch

                # Preprocess
                with profilers[0]:
                    im = self.preprocess(im0s)

                # Inference
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)

                # Postprocess
                with profilers[2]:
                    if isinstance(self.model, AutoBackend):
                        self.results = self.postprocess(preds, im, im0s)
                    else:
                        self.results = self.model.postprocess(path, preds, im, im0s)
                

            
                self.run_callbacks('on_predict_postprocess_end')
                # Visualize, save, write results
                
                n = len(im0s)
                cls= self.results[0].boxes.cls
                id=self.results[0].boxes.id
                bbox=self.results[0].boxes.xyxy
                # cls 값이 0인 바운딩 
                bbox_cls_0 = [bbox[i] for i in range(len(cls)) if cls[i] == 0]
                bbox_cls_1 = [bbox[i] for i in range(len(cls)) if cls[i] == 1]

                bbox_id_0, bbox_id_1 = [], []

                if id is not None and cls is not None:
                    # cls 값이 0인 바운딩 박스의 id
                    bbox_id_0 = [id[i] for i in range(len(cls)) if cls[i] == 0]
                    # cls 값이 1인 바운딩 박스의 id
                    bbox_id_1 = [id[i] for i in range(len(cls)) if cls[i] == 1]
                else:
                    # Handle the case where id or cls is None (e.g., print an error message)
                    print("Warning: 'id' or 'cls' is None. Bbox filtering skipped.")



                matches = group_pairs(bbox_cls_0, bbox_cls_1, bbox_id_0, bbox_id_1)
                
                #video inference때 디버깅용으로 정의
                # 현재 프레임수
                # frame=self.dataset.frame
               
                # if frame==930:

                #     ii=1
                # #카운팅기반 코드 

                self.update_group_pair_counts(matches)
                img_count=self.dataset.count
          
                #프레임의 person,group매칭 id들을 새로운 곳에 저장을하고 새로 하나씩 id를 부여하는 코드
                
                self.all_group= []
             
                self.all_group.append(self.ori_group_id)
                self.all_group.append(self.new_ori_group_id)
       
                ########################
           

               
                for i in range(n):
                    self.seen += 1
                    self.results[i].speed = {
                        'preprocess': profilers[0].dt * 1E3 / n,
                        'inference': profilers[1].dt * 1E3 / n,
                        'postprocess': profilers[2].dt * 1E3 / n}
                    p, im0 = path[i], None if self.source_type.tensor else im0s[i].copy()
                    p = Path(p)
            
                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        s += self.write_results(i, self.results, (p, im, im0))
                    if self.args.save or self.args.save_txt:
                        self.results[i].save_dir = self.save_dir.__str__()
                    if self.args.show and self.plotted_img is not None:
                        self.show(p)
                    ###show-labels가 저장되는곳
                    if self.args.save and self.plotted_img is not None:
                        self.save_preds(vid_cap, i, str(self.save_dir / p.name))
        

                        

                self.run_callbacks('on_predict_batch_end')
                yield from self.results

                # Print time (inference-only)
                if self.args.verbose:
                    LOGGER.info(f'{s}{profilers[1].dt * 1E3:.1f}ms')

        # Release assets
        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()  # release final video writer

        # Print results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1E3 for x in profilers)  # speeds per image
            LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
                        f'{(1, 3, *im.shape[2:])}' % t)
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob('labels/*.txt')))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        self.run_callbacks('on_predict_end')
  
    


    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        self.model = AutoBackend(model or self.args.model,
                                 device=select_device(self.args.device, verbose=verbose),
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=verbose)

        self.device = self.model.device  # update device
        self.args.half = self.model.fp16  # update half
        self.model.eval()

    def show(self, p):
        """Display an image in a window using OpenCV imshow()."""
        im0 = self.plotted_img
        if platform.system() == 'Linux' and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(500 if self.batch[3].startswith('image') else 1)  # 1 millisecond



    def save_txt(self, cls, txt_file='save.txt', save_conf=False):

    
        if cls:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
            with open(txt_file, 'a') as f:
                f.writelines(' '.join([str(c) for c in cls]) + '\n')

    def save_preds(self, vid_cap, idx, save_path):
        
        
        """Save video predictions as mp4 at specified path."""
        im0 = self.plotted_img
        # Save imgs
        if self.dataset.mode == 'image':
            cv2.imwrite(save_path, im0)
        else:  # 'video' or 'stream'
            frames_path = f'{save_path.split(".", 1)[0]}_frames/'
            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                    self.vid_frame[idx] = 0
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                suffix, fourcc = ('.mp4', 'avc1') if MACOS else ('.avi', 'WMV2') if WINDOWS else ('.avi', 'MJPG')
                #여기서 track.py 파일의 for문으로 넘어감
                self.vid_writer[idx] = cv2.VideoWriter(str(Path(save_path).with_suffix(suffix)),
                                                       cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
               
            # Write video
            self.vid_writer[idx].write(im0)
        
            # Write frame
            if self.args.save_frames:
                cv2.imwrite(f'{frames_path}{self.vid_frame[idx]}.jpg', im0)
                self.vid_frame[idx] += 1

    def run_callbacks(self, event: str):
        """Runs all registered callbacks for a specific event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func):
        """Add callback."""
        self.callbacks[event].append(func)


